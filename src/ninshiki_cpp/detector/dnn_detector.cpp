// Copyright (c) 2021 ICHIRO ITS
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "ninshiki_cpp/detector/dnn_detector.hpp"

#include <map>
#include <string>
#include <chrono>
#include <vector>
#include <nlohmann/json.hpp>

#include "jitsuyo/config.hpp"
#include "jitsuyo/linux.hpp"

namespace ninshiki_cpp
{
namespace detector
{

DnnDetector::DnnDetector()
{
}

void DnnDetector::load_configuration(const std::string & path)
{
  std::string ss = config_path + "/dnn_model.json";

  nlohmann::json dnn_config;
  if (!jitsuyo::load_config(path, "/dnn_model.json", dnn_config)) {
    std::cerr << "Failed to load dnn configuration file" << std::endl;
    return;
  }

  for (auto & item : dnn_config.items()) {
    try {
      if (item.key() == "model") {
        model_path = static_cast<std::string>(getenv("HOME")) + item.value().get<std::string>();
      } else if (item.key() == "config") {
        config = static_cast<std::string>(getenv("HOME")) + item.value().get<std::string>();
      } else if (item.key() == "classes") {
        file_name = static_cast<std::string>(getenv("HOME")) + item.value().get<std::string>();
      }
    } catch (nlohmann::json::parse_error & ex) {
      std::cerr << "parse error at byte " << ex.byte << std::endl;
    }
  }

  model_suffix = jitsuyo::split_string(model_path, ".");

  if (model_suffix == "weights") {
    net = cv::dnn::readNet(model_path, config, "");
  } else if (model_suffix == "xml") {
    // OpenVINO initialization is deferred to set_computation_method()
    // so that the device selection (GPU/MYRIAD/CPU) is available.
  } else {
    throw std::runtime_error("Model suffix is not supported");
  }

  std::ifstream ifs(file_name.c_str());
  std::string line;
  while (std::getline(ifs, line)) {
    classes.push_back(line);
  }

  iterations = 0;

}

void DnnDetector::set_computation_method(bool gpu, bool myriad)
{
  this->gpu = gpu;
  this->myriad = myriad;

  if (this->model_suffix == "xml") {
    std::string device = "CPU";
    if (gpu) {
      device = "GPU";
    } else if (myriad) {
      device = "MYRIAD";
    }
    initialize_openvino(device);
    return;
  }

  // Set computation method (gpu, myriad, or CPU)
  if (gpu) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  } else if (myriad) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
  } else {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }

  // Cache layer metadata once — avoid repeated graph introspection on every detection
  if (this->model_suffix == "weights") {
    layer_output = net.getUnconnectedOutLayersNames();
    out_layers = net.getUnconnectedOutLayers();
    out_layer_type = net.getLayer(out_layers[0])->type;
  }
}

void DnnDetector::initialize_openvino(const std::string & device)
{
  ov::Core core;
  std::shared_ptr<ov::Model> model = core.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

  ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
  ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255.0f, 255.0f, 255.0f });
  ppp.input().model().set_layout("NCHW");
  ppp.output().tensor().set_element_type(ov::element::f32);
  model = ppp.build();

  ov::AnyMap device_config = {
    ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
    ov::hint::enable_cpu_pinning(false),
    // inference_num_threads only can be used in cpu mode
    // ov::inference_num_threads(1)
  };
  
  this->compiled_model = core.compile_model(model, device, device_config);
  this->infer_request = compiled_model.create_infer_request();

  // Map pre-allocated tensor memory once for zero-copy
  input_tensor = infer_request.get_input_tensor();
  this->tensor_mat = cv::Mat(320, 320, CV_8UC3, input_tensor.data<uint8_t>());

  // Register async callback
  infer_request.set_callback([this](std::exception_ptr ex) {
    if (ex) {
      std::cerr << "OpenVINO inference exception!" << std::endl;
      this->is_inferencing = false;
      return;
    }
    this->postprocess_ir();
  });
}

void DnnDetector::detection(const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  if (model_suffix == "weights") {
    detect_darknet(image, conf_threshold, nms_threshold);
  } else if (model_suffix == "xml") {
    detect_ir(image, conf_threshold, nms_threshold);
    return;
  } else {
    detect_tensorflow(image, conf_threshold, nms_threshold);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> latency = end_time - start_time;
  total_latency += latency.count();
  iterations++;

  printf("Inference time: %.2f ms, %d\n", latency.count(), iterations);
  printf("Average latency: %.2f ms\n", total_latency / iterations);
  printf("--------------------------------\n");
}

void DnnDetector::detect_darknet(const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  // Create a 4D blob from a frame
  static cv::Mat blob;
  cv::Size input_size = cv::Size(320, 320);
  cv::dnn::blobFromImage(image, blob, 1.0, input_size, cv::Scalar(), false, false, CV_8U);

  net.setInput(blob, "", 0.00392, cv::Scalar(0, 0, 0, 0));
  std::vector<cv::Mat> outs;
  net.forward(outs, layer_output);

  // Get width and height from image
  img_width = static_cast<double>(image.cols);
  img_height = static_cast<double>(image.rows);

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect2d> boxes;

  if (out_layer_type == "Region") {
    for (size_t i = 0; i < outs.size(); ++i) {
      // Network produces output blob with a shape NxC where N is a number of
      // detected objects and C is a number of classes + 4 where the first 4
      // numbers are [center_x, center_y, width, height]
      float * data = reinterpret_cast<float *>(outs[i].data);
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        cv::Point class_id_point;
        double confidence;
        cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
        if (confidence > conf_threshold) {
          double centerX = data[0] * img_width;
          double centerY = data[1] * img_height;
          double width = data[2] * img_width;
          double height = data[3] * img_height;
          double left = centerX - width / 2;
          double top = centerY - height / 2;

          class_ids.push_back(class_id_point.x);
          confidences.push_back(static_cast<float>(confidence));
          boxes.push_back(cv::Rect2d(left, top, width, height));
        }
      }
    }
  }

  // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends
  // we need NMS in sample or NMS is required if number of outputs > 1
  if (out_layers.size() > 1 || (out_layer_type == "Region" && (this->gpu || this->myriad))) {
    std::map<int, std::vector<size_t>> class2indices;
    for (size_t i = 0; i < class_ids.size(); i++) {
      if (confidences[i] >= conf_threshold) {
        class2indices[class_ids[i]].push_back(i);
      }
    }
    std::vector<cv::Rect2d> nms_boxes;
    std::vector<float> nms_confidences;
    std::vector<int> nms_class_ids;
    for (std::map<int, std::vector<size_t>>::iterator it = class2indices.begin();
      it != class2indices.end(); ++it)
    {
      std::vector<cv::Rect2d> local_boxes;
      std::vector<float> local_confidences;
      std::vector<size_t> class_indices = it->second;
      for (size_t i = 0; i < class_indices.size(); i++) {
        local_boxes.push_back(boxes[class_indices[i]]);
        local_confidences.push_back(confidences[class_indices[i]]);
      }
      std::vector<int> nmsIndices;
      cv::dnn::NMSBoxes(local_boxes, local_confidences, conf_threshold, nms_threshold, nmsIndices);
      for (size_t i = 0; i < nmsIndices.size(); i++) {
        size_t idx = nmsIndices[i];
        nms_boxes.push_back(local_boxes[idx]);
        nms_confidences.push_back(local_confidences[idx]);
        nms_class_ids.push_back(it->first);
      }
    }
    boxes = nms_boxes;
    class_ids = nms_class_ids;
    confidences = nms_confidences;
  }

  if (boxes.size()) {
    for (size_t i = 0; i < boxes.size(); ++i) {
      cv::Rect box = boxes[i];
      if (box.width * box.height != 0) {
        // Add detected object into vector
        ninshiki_interfaces::msg::DetectedObject detection_object;
        detection_object.label = classes[class_ids[i]];
        detection_object.score = confidences[i];
        detection_object.left = box.x;
        detection_object.top = box.y;
        detection_object.right = box.width;
        detection_object.bottom = box.height;

        detection_result.detected_objects.push_back(detection_object);
      }
    }
  }
}

void DnnDetector::detect_tensorflow(
  const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  static cv::Mat blob;
  cv::Size input_size = cv::Size(300, 300);
  cv::dnn::blobFromImage(image, blob, 1.0, input_size, cv::Scalar(), true, false, CV_8U);

  net.setInput(blob);
  cv::Mat output = net.forward();
  cv::Mat detection_mat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect2f> boxes;

  // Get width and height from image
  img_width = static_cast<double>(image.cols);
  img_height = static_cast<double>(image.rows);

  // check detection_mat not NULL
  if (detection_mat.cols * detection_mat.rows != 0) {
    for (int i = 0; i < detection_mat.rows; i++) {
      int class_id = detection_mat.at<float>(i, 1);
      float confidence = detection_mat.at<float>(i, 2);

      // Check if the detection is of good quality
      if (confidence > conf_threshold) {
        float left = detection_mat.at<float>(i, 3) * img_width;
        float top = detection_mat.at<float>(i, 4) * img_height;
        float width = detection_mat.at<float>(i, 5) * img_width - left;
        float height = detection_mat.at<float>(i, 6) * img_height - top;

        class_ids.push_back(class_id - 1);
        confidences.push_back(confidence);
        boxes.push_back(cv::Rect2f(left, top, width, height));
      }
    }
  }

  if (boxes.size()) {
    for (size_t i = 0; i < boxes.size(); ++i) {
      cv::Rect box = boxes[i];
      if (box.width * box.height != 0) {
        // Add detected object into vector
        ninshiki_interfaces::msg::DetectedObject detection_object;
        detection_object.label = classes[class_ids[i]];
        detection_object.score = confidences[i];
        detection_object.left = box.x / img_width;
        detection_object.top = box.y / img_height;
        detection_object.right = (box.x + box.width) / img_width;
        detection_object.bottom = (box.y + box.height) / img_height;

        detection_result.detected_objects.push_back(detection_object);
      }
    }
  }
}

void DnnDetector::postprocess_ir()
{
  // Postprocessing
  std::vector<cv::Rect> boxes;
  std::vector<int> class_ids;
  std::vector<float> confidences;

  const ov::Tensor& output_tensor = infer_request.get_output_tensor();
  ov::Shape output_shape = output_tensor.get_shape();
  float* detections = output_tensor.data<float>();

  int out_rows = output_shape[1];
  int out_cols = output_shape[2];
  const cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)detections);

  for (int i = 0; i < det_output.cols; ++i) {
    const cv::Mat classes_scores = det_output.col(i).rowRange(4, classes.size() + 4);
    cv::Point class_id_point;
    double score;
    cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

    if (score > conf_threshold) {
      const float cx = det_output.at<float>(0, i);
      const float cy = det_output.at<float>(1, i);
      const float ow = det_output.at<float>(2, i);
      const float oh = det_output.at<float>(3, i);
      cv::Rect box;
      box.x = static_cast<int>((cx - 0.5 * ow) );
      box.y = static_cast<int>((cy - 0.5 * oh));
      box.width = static_cast<int>(ow);
      box.height = static_cast<int>(oh);

      boxes.push_back(box);
      class_ids.push_back(class_id_point.y);
      confidences.push_back(score);
    }
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

  std::lock_guard<std::mutex> lock(result_mutex);
  async_detection_result.detected_objects.clear();

  for (int i = 0; i < nms_result.size(); ++i) {
    int idx = nms_result[i];
    if (class_ids[idx] == 6) {
      continue;
    }

    ninshiki_interfaces::msg::DetectedObject detection_object;
    
    detection_object.label = classes[class_ids[idx]];
    detection_object.score = confidences[idx];
    detection_object.left = boxes[idx].x * rx;
    detection_object.top = boxes[idx].y * ry;
    detection_object.right = boxes[idx].width * rx;
    detection_object.bottom = boxes[idx].height * ry;

    async_detection_result.detected_objects.push_back(detection_object);
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> latency = end_time - openvino_start_time;
  total_latency += latency.count();
  iterations++;

  printf("Inference time: %.2f ms, %d\n", latency.count(), iterations);
  printf("Average latency: %.2f ms\n", total_latency / iterations);
  printf("--------------------------------\n");

  is_inferencing = false;  // set under result_mutex (postprocess done, no more writer conflict)
}

void DnnDetector::detect_ir(const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  // Check and set is_inferencing under inference_mutex only
  {
    std::lock_guard<std::mutex> lock(inference_mutex);

    // Brief copy of the latest result — needs result_mutex too
    {
      std::lock_guard<std::mutex> lock_result(result_mutex);
      detection_result = async_detection_result;
    }

    if (is_inferencing) {
      return;  // Drop this frame, return stale result
    }

    is_inferencing = true;
  }

  // Preprocessing runs WITHOUT holding any lock — no deadlock risk
  this->conf_threshold = conf_threshold;
  this->nms_threshold = nms_threshold;

  try {
    img_width = static_cast<double>(image.cols);
    img_height = static_cast<double>(image.rows);

    cv::Size new_shape = cv::Size(320, 320);

    float ratio = float(new_shape.width / (img_width > img_height ? img_width : img_height));
    int new_unpadW = int(round(img_width * ratio));
    int new_unpadH = int(round(img_height * ratio));

    cv::resize(image, this->resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;

    // Copy directly into OpenVINO shared tensor memory
    cv::copyMakeBorder(this->resized_image, this->tensor_mat, 0, dh, 0, dw, cv::BORDER_CONSTANT, cv::Scalar(100, 100, 100));

    this->rx = (float)image.cols / (float)(new_shape.width - dw);
    this->ry = (float)image.rows / (float)(new_shape.height - dh);
  } catch (const std::exception& e) {
    std::cerr << "OpenVINO preprocessing exception: " << e.what() << std::endl;
    std::lock_guard<std::mutex> lock(inference_mutex);
    is_inferencing = false;
    return;
  } catch (...) {
    std::cerr << "OpenVINO unknown preprocessing exception" << std::endl;
    std::lock_guard<std::mutex> lock(inference_mutex);
    is_inferencing = false;
    return;
  }

  // Start Asynchronous Inference
  openvino_start_time = std::chrono::high_resolution_clock::now();
  infer_request.start_async();
}

}  // namespace detector
}  // namespace ninshiki_cpp
