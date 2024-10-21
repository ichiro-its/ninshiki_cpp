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

namespace ninshiki_cpp::detector
{

class DnnDetector : public Detector
{
public:
  explicit DnnDetector();

  void set_computation_method(bool gpu = false, bool myriad = false);
  void detection(const cv::Mat & image, float conf_threshold, float nms_threshold);
  void detect_darknet(const cv::Mat & image, float conf_threshold, float nms_threshold);
  void detect_tensorflow(const cv::Mat & image, float conf_threshold, float nms_threshold);
  void detect_ir(const cv::Mat & image, float conf_threshold, float nms_threshold);
  void initialize_openvino(const std::string & model_path);

private:
  std::string file_name;
  std::string model_suffix;
  std::vector<std::string> classes;

  bool gpu;
  bool myriad;

  cv::dnn::Net net;

  ov::Tensor input_tensor;
  ov::InferRequest infer_request;
  ov::CompiledModel compiled_model;

  float rx, ry;

  int iterations;
  double avg_latency;
  double total_latency;
};

}  // namespace ninshiki_cpp::detector

#endif  // NINSHIKI_CPP__DETECTOR__DNN_DETECTOR_HPP_
