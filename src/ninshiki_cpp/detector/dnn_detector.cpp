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
  printf("[initialize_openvino] device=%s, model_path=%s\n", device.c_str(), model_path.c_str());
  core = ov::Core();
  std::shared_ptr<ov::Model> model = core.read_model(model_path);
  ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

  ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
  ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255.0f, 255.0f, 255.0f });
  ppp.input().model().set_layout("NCHW");
  ppp.output().tensor().set_element_type(ov::element::f32);
  model = ppp.build();

  ov::AnyMap device_config = {
    ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
    ov::hint::enable_cpu_pinning(false),
  };

  compiled_model = core.compile_model(model, device, device_config);

  // Single request for now — multi-request pipeline next
  infer_requests.reserve(1);
  preprocess_data.reserve(1);
  request_pending.reserve(1);
  infer_requests.push_back(compiled_model.create_infer_request());
  preprocess_data.push_back({});
  request_pending.push_back(false);

  infer_requests[0].set_callback([this](std::exception_ptr ex) mutable {
    if (ex) {
      std::cerr << "OpenVINO inference exception on request 0!" << std::endl;
      std::lock_guard<std::mutex> lock(this->pending_mutex);
      this->request_pending[0] = false;
      return;
    }
    auto pre = this->preprocess_data[0];
    this->postprocess_ir(0, pre);
  });

  printf("[initialize_openvino] request created successfully\n");
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

void DnnDetector::postprocess_ir(size_t req_idx, const PreprocessData & pre)
{
  const ov::Tensor & output_tensor = infer_requests[req_idx].get_output_tensor();
  ov::Shape output_shape = output_tensor.get_shape();
  float * detections = output_tensor.data<float>();

  int out_rows = static_cast<int>(output_shape[1]);
  int out_cols = static_cast<int>(output_shape[2]);
  const cv::Mat det_output(out_rows, out_cols, CV_32F, detections);

  std::vector<cv::Rect> boxes;
  std::vector<int> class_ids;
  std::vector<float> confidences;

  for (int i = 0; i < det_output.cols; ++i) {
    const cv::Mat classes_scores = det_output.col(i).rowRange(4, classes.size() + 4);
    cv::Point class_id_point;
    double score;
    cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

    if (score > pre.conf_threshold) {
      const float cx = det_output.at<float>(0, i);
      const float cy = det_output.at<float>(1, i);
      const float ow = det_output.at<float>(2, i);
      const float oh = det_output.at<float>(3, i);
      boxes.emplace_back(
        static_cast<int>((cx - 0.5 * ow)),
        static_cast<int>((cy - 0.5 * oh)),
        static_cast<int>(ow),
        static_cast<int>(oh));
      class_ids.push_back(class_id_point.y);
      confidences.push_back(static_cast<float>(score));
    }
  }

  std::vector<int> nms_result;
  cv::dnn::NMSBoxes(boxes, confidences, pre.conf_threshold, pre.nms_threshold, nms_result);

  {
    std::lock_guard<std::mutex> lock(result_mutex);
    async_detection_result.detected_objects.clear();

    for (int nms_idx : nms_result) {
      if (class_ids[nms_idx] == 6) {
        continue;
      }
      ninshiki_interfaces::msg::DetectedObject detection_object;
      detection_object.label = classes[class_ids[nms_idx]];
      detection_object.score = confidences[nms_idx];
      detection_object.left = boxes[nms_idx].x * pre.rx;
      detection_object.top = boxes[nms_idx].y * pre.ry;
      detection_object.right = boxes[nms_idx].width * pre.rx;
      detection_object.bottom = boxes[nms_idx].height * pre.ry;
      async_detection_result.detected_objects.push_back(detection_object);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> latency = end_time - pre.start_time;
  total_latency += latency.count();
  iterations++;

  auto preprocess_duration = pre.preprocess_end_time - pre.start_time;
  auto inference_duration = latency;
  printf("[Req %zu] preprocess: %.2f ms  |  inference: %.2f ms  |  total: %.2f ms\n",
    req_idx, preprocess_duration.count(), inference_duration.count(),
    (preprocess_duration + inference_duration).count());
  printf("Average latency: %.2f ms\n", total_latency / iterations);

  std::lock_guard<std::mutex> lock(pending_mutex);
  request_pending[req_idx] = false;
}

void DnnDetector::detect_ir(const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  printf("[detect_ir] ENTER, rows=%d cols=%d\n", image.rows, image.cols);

  // Guard: ensure initialize_openvino() was called
  if (infer_requests.empty()) {
    std::cerr << "detect_ir called but no infer requests initialized!" << std::endl;
    return;
  }

  // If previous inference still running, return stale result
  {
    std::lock_guard<std::mutex> lock(pending_mutex);
    if (request_pending[0]) {
      printf("[detect_ir] still running, returning stale\n");
      std::lock_guard<std::mutex> lock_result(result_mutex);
      detection_result = async_detection_result;
      return;
    }
    request_pending[0] = true;
  }

  // Brief copy of the latest result
  {
    std::lock_guard<std::mutex> lock(result_mutex);
    detection_result = async_detection_result;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  // Store preprocessing data for callback
  preprocess_data[0].conf_threshold = conf_threshold;
  preprocess_data[0].nms_threshold = nms_threshold;
  preprocess_data[0].start_time = start_time;

  img_width = static_cast<double>(image.cols);
  img_height = static_cast<double>(image.rows);

  // Letterbox resize to 320x320
  cv::Size model_input(320, 320);
  float ratio = static_cast<float>(model_input.width / (img_width > img_height ? img_width : img_height));
  int new_unpadW = static_cast<int>(round(img_width * ratio));
  int new_unpadH = static_cast<int>(round(img_height * ratio));
  int dw = model_input.width - new_unpadW;
  int dh = model_input.height - new_unpadH;
  preprocess_data[0].rx = static_cast<float>(img_width) / static_cast<float>(model_input.width - dw);
  preprocess_data[0].ry = static_cast<float>(img_height) / static_cast<float>(model_input.height - dh);

  // Resize image
  cv::Mat resized_320;
  cv::resize(image, resized_320, model_input, 0, 0, cv::INTER_AREA);

  if (!resized_320.data) {
    std::cerr << "cv::resize failed!" << std::endl;
    std::lock_guard<std::mutex> lock(pending_mutex);
    request_pending[0] = false;
    return;
  }

  preprocess_data[0].preprocess_end_time = std::chrono::high_resolution_clock::now();

  printf("[detect_ir] starting async\n");

  // Set input tensor and start async
  ov::Tensor input_tensor(
    ov::element::u8,
    ov::Shape{1, 320, 320, 3},
    resized_320.data);
  infer_requests[0].set_input_tensor(input_tensor);
  infer_requests[0].start_async();
}

}  // namespace detector
}  // namespace ninshiki_cpp
