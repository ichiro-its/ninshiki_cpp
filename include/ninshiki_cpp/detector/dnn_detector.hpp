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

#ifndef NINSHIKI_CPP__DETECTOR__DNN_DETECTOR_HPP_
#define NINSHIKI_CPP__DETECTOR__DNN_DETECTOR_HPP_

#include "ninshiki_cpp/detector/detector.hpp"
#include "ninshiki_interfaces/msg/detected_object.hpp"
#include "ninshiki_interfaces/msg/detected_objects.hpp"

#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>

namespace ninshiki_cpp::detector
{

class DnnDetector : public Detector
{
public:
  explicit DnnDetector();

  void set_computation_method(bool gpu = false, bool myriad = false);
  void load_configuration(const std::string & path);
  void detection(const cv::Mat & image, float conf_threshold, float nms_threshold);
  void detect_darknet(const cv::Mat & image, float conf_threshold, float nms_threshold);
  void detect_tensorflow(const cv::Mat & image, float conf_threshold, float nms_threshold);
  void detect_ir(const cv::Mat & image, float conf_threshold, float nms_threshold);
  void initialize_openvino(const std::string & device);

private:
  std::string model_path;
  std::string config;
  std::string file_name;
  std::string model_suffix;
  std::vector<std::string> classes;

  bool gpu;
  bool myriad;

  cv::dnn::Net net;

  // Cached network layer metadata — initialized once, reused every detection
  std::vector<cv::String> layer_output;
  std::vector<int> out_layers;
  std::string out_layer_type;

  ov::CompiledModel compiled_model;

  // Async state — multi-request pipeline
  static constexpr size_t NUM_CONCURRENT_REQUESTS = 3;
  std::vector<ov::InferRequest> infer_requests;
  std::vector<ov::Tensor> input_tensors;       // per-request input tensors
  std::vector<cv::Mat> tensor_mats;             // per-request pre-allocated buffers
  std::vector<cv::Mat> resized_images;           // per-request resize buffers
  size_t request_idx = 0;                         // circular index

  struct PreprocessData
  {
    float conf_threshold;
    float nms_threshold;
    float rx, ry;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> preprocess_end_time;
  };
  std::vector<PreprocessData> preprocess_data;
  std::vector<bool> request_pending;  // which requests are in-flight
  std::mutex pending_mutex;            // guards request_pending

  void postprocess_ir(size_t req_idx, const PreprocessData & pre);

  std::mutex result_mutex;             // guards async_detection_result
  ninshiki_interfaces::msg::DetectedObjects async_detection_result;

  int iterations;
  double avg_latency;
  double total_latency;
};

}  // namespace ninshiki_cpp::detector

#endif  // NINSHIKI_CPP__DETECTOR__DNN_DETECTOR_HPP_
