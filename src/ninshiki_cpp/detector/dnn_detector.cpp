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

#include <openvino/runtime/intel_gpu/properties.hpp>

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

  inference_count = 0;
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

void DnnDetector::set_nms_free(bool nms_free)
{
  this->nms_free = nms_free;
}

void DnnDetector::initialize_openvino(const std::string & device)
{
  printf("[initialize_openvino] device=%s, model_path=%s\n", device.c_str(), model_path.c_str());

  core = ov::Core();
  std::shared_ptr<ov::Model> model = core.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

  ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
  ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255.0f, 255.0f, 255.0f });
  ppp.input().model().set_layout("NCHW");
  ppp.output().tensor().set_element_type(ov::element::f32);
  model = ppp.build();

  ov::AnyMap device_config;
  if (device == "GPU") {
    // Intel Iris Xe / integrated GPU — throughput mode + high priority host tasks
    device_config = {
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::intel_gpu::hint::host_task_priority(ov::hint::Priority::HIGH),
      ov::intel_gpu::hint::queue_priority(ov::hint::Priority::HIGH),
      ov::intel_gpu::hint::queue_throttle(ov::intel_gpu::hint::ThrottleLevel::LOW),
    };
  } else if (device == "CPU") {
    // x86 CPU — throughput mode with NUMA awareness
    device_config = {
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
      ov::num_streams(1),
    };
  } else {
    device_config = {
      ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
    };
  }

  compiled_model = core.compile_model(model, device, device_config);
  infer_request = compiled_model.create_infer_request();

  printf("[initialize_openvino] openvino initialized successfully\n");
}

void DnnDetector::detection(const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  if (model_suffix == "weights") {
    detect_darknet(image, conf_threshold, nms_threshold);
  } else if (model_suffix == "xml") {
    detect_ir(image, conf_threshold, nms_threshold);
  } else {
    detect_tensorflow(image, conf_threshold, nms_threshold);
  }

  // Print Inference Time
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> latency = end_time - start_time;
  total_latency += latency.count();
  inference_count++;

  std::cout << "[Frame " << inference_count << "] Total time: " << latency.count() << " ms | ";
  std::cout << "Average latency: " << (total_latency / inference_count) << " ms" << std::endl;
  std::cout << "--------------------------------" << std::endl;
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

void DnnDetector::detect_ir(const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  if (!infer_request) {
    std::cerr << "detect_ir called but infer request not initialized!" << std::endl;
    return;
  }

  // Preprocessing
  auto start_time = std::chrono::high_resolution_clock::now();
  std::chrono::time_point<std::chrono::high_resolution_clock> preprocess_end_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> inference_end_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> postprocess_end_time;

  try {
    img_width = static_cast<double>(image.cols);
    img_height = static_cast<double>(image.rows);

    cv::Size new_shape(320, 320);
    cv::Mat resized_image;

    float ratio = static_cast<float>(new_shape.width) / static_cast<float>(img_width > img_height ? img_width : img_height);
    int new_unpadW = static_cast<int>(round(img_width * ratio));
    int new_unpadH = static_cast<int>(round(img_height * ratio));

    cv::resize(image, resized_image, cv::Size(new_unpadW, new_unpadH), 0, 0, cv::INTER_AREA);
    int dw = new_shape.width - new_unpadW;
    int dh = new_shape.height - new_unpadH;

    // Apply letterbox padding
    cv::Mat resized_320;
    cv::copyMakeBorder(resized_image, resized_320, 0, dh, 0, dw, cv::BORDER_CONSTANT, cv::Scalar(100, 100, 100));

    rx = static_cast<float>(img_width) / static_cast<float>(resized_320.cols - dw);
    ry = static_cast<float>(img_height) / static_cast<float>(resized_320.rows - dh);

    // Create tensor using model's actual input shape
    ov::Shape input_shape = compiled_model.input().get_shape();
    ov::Tensor input_tensor(ov::element::u8, input_shape, resized_320.data);
    infer_request.set_input_tensor(input_tensor);

    preprocess_end_time = std::chrono::high_resolution_clock::now();
  } catch (const std::exception& e) {
    std::cerr << "exception: " << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "unknown exception" << std::endl;
  }

  // Inference
  infer_request.infer();
  inference_end_time = std::chrono::high_resolution_clock::now();
  const ov::Tensor & output_tensor = infer_request.get_output_tensor();
  ov::Shape output_shape = output_tensor.get_shape();
  auto detections = output_tensor.data<float>();

  // Postprocessing
  if (nms_free) {
    // No NMS needed if model handles it internally
    int num_detections = static_cast<int>(output_shape[1]);
    int num_cols = static_cast<int>(output_shape[2]);  // should be 6
    const cv::Mat det_output(num_detections, num_cols, CV_32F, const_cast<float*>(detections));

    auto & out = detection_result.detected_objects;
    out.reserve(num_detections);

    for (int i = 0; i < num_detections; ++i) {
      const float* row = det_output.ptr<float>(i);

      float conf = row[4];
      int class_id = static_cast<int>(row[5]);

      if (conf < conf_threshold || class_id == 6) continue;

      float x1 = row[0];
      float y1 = row[1];
      float x2 = row[2];
      float y2 = row[3];

      auto& obj = out.emplace_back();
      obj.label = classes[class_id];
      obj.score = conf;
      obj.left = static_cast<int>(x1) * rx;
      obj.top = static_cast<int>(y1) * ry;
      obj.right = static_cast<int>(x2 - x1) * rx;
      obj.bottom = static_cast<int>(y2 - y1) * ry;
    }
  } else {
    int out_rows = static_cast<int>(output_shape[1]);
    int out_cols = static_cast<int>(output_shape[2]);
    const cv::Mat det_output(out_rows, out_cols, CV_32F, const_cast<float*>(detections));

    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    boxes.reserve(out_cols);
    class_ids.reserve(out_cols);
    confidences.reserve(out_cols);

    const float* data = reinterpret_cast<const float*>(det_output.data);

    for (int i = 0; i < out_cols; ++i) {
      float cx = data[0 * out_cols + i];
      float cy = data[1 * out_cols + i];
      float ow = data[2 * out_cols + i];
      float oh = data[3 * out_cols + i];

      int best_class = -1;
      float best_score = -1.0f;

      for (int c = 0; c < static_cast<int>(classes.size()); ++c) {
        float score = data[(4 + c) * out_cols + i];
        if (score > best_score) {
          best_score = score;
          best_class = c;
        }
      }

      if (best_score < conf_threshold) continue;

      boxes.emplace_back(
        static_cast<int>(cx - 0.5f * ow),
        static_cast<int>(cy - 0.5f * oh),
        static_cast<int>(ow),
        static_cast<int>(oh)
      );

      class_ids.emplace_back(best_class);
      confidences.emplace_back(best_score);
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    auto& out = detection_result.detected_objects;
    out.reserve(nms_result.size());

    for (int idx : nms_result) {
      int class_id = class_ids[idx];
      if (class_id == 6) continue;

      const auto& box = boxes[idx];

      auto& obj = out.emplace_back();
      obj.label = classes[class_id];
      obj.score = confidences[idx];
      obj.left = box.x * rx;
      obj.top = box.y * ry;
      obj.right = box.width * rx;
      obj.bottom = box.height * ry;
    }
  }

  postprocess_end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> preprocess_duration = preprocess_end_time - start_time;
  std::chrono::duration<double, std::milli> inference_duration = inference_end_time - preprocess_end_time;
  std::chrono::duration<double, std::milli> postprocess_duration = postprocess_end_time - inference_end_time;

  std::cout << "Preprocessing time: " << preprocess_duration.count() << " ms" << std::endl;
  std::cout << "Inference time: " << inference_duration.count() << " ms" << std::endl;
  std::cout << "Postprocessing time: " << postprocess_duration.count() << " ms" << std::endl;
}

}  // namespace detector
}  // namespace ninshiki_cpp
