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

#ifndef NINSHIKI_CPP__DETECTOR__YOLO_HPP_
#define NINSHIKI_CPP__DETECTOR__YOLO_HPP_

#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#include "ninshiki_interfaces/msg/detected_object.hpp"

namespace ninshiki_cpp::detector
{

class Yolo
{
public:
  std::vector<ninshiki_interfaces::msg::DetectedObject> detection_result;

  Yolo(bool gpu = false, bool myriad = false);
  void pass_image_to_network(cv::Mat image);
  void detection(float conf_threshold, float nms_threshold);

private:
  std::string file_name;
  std::vector<std::string> classes;

  bool gpu;
  bool myriad;

  cv::dnn::Net net;
  std::vector<cv::Mat> outs;

  int width;
  int height;
};

}  // namespace ninshiki_cpp::detector

#endif  // NINSHIKI_CPP__DETECTOR__YOLO_HPP_