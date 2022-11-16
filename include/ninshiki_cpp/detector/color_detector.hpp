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

#ifndef NINSHIKI_CPP__DETECTOR__COLOR_DETECTOR_HPP_
#define NINSHIKI_CPP__DETECTOR__COLOR_DETECTOR_HPP_

#include <opencv2/opencv.hpp>
#include <cmath>

#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

#include "keisan/geometry/point_2.hpp"
#include "ninshiki_interfaces/msg/point.hpp"
#include "ninshiki_interfaces/msg/contour.hpp"
#include "ninshiki_interfaces/msg/contours.hpp"
#include "ninshiki_cpp/utils/color.hpp"
#include "ninshiki_cpp/utils/utils.hpp"
#include "ninshiki_cpp/detector/detector.hpp"

namespace ninshiki_cpp::detector
{

class ColorDetector : public Detector
{
public:
  ColorDetector();
  ~ColorDetector();

  bool load_configuration(const std::string & path);
  bool load_configuration() {load_configuration(config_path);}
  bool save_configuration();
  bool sync_configuration();

  cv::Mat classify(cv::Mat input);
  cv::Mat classify_gray(cv::Mat input);

  int get_min_hue() {return min_hue;}
  int get_max_hue() {return max_hue;}
  int get_min_saturation() {return min_saturation;}
  int get_max_saturation() {return max_saturation;}
  int get_min_value() {return min_value;}
  int get_max_value() {return max_value;}

  void set_min_hue(int value) {min_hue = keisan::clamp(value, 0, 360);}
  void set_max_hue(int value) {max_hue = keisan::clamp(value, 0, 360);}
  void set_min_saturation(int value) {min_saturation = keisan::clamp(value, 0, 100);}
  void set_max_saturation(int value) {max_saturation = keisan::clamp(value, 0, 100);}
  void set_min_value(int value) {min_value = keisan::clamp(value, 0, 100);}
  void set_max_value(int value) {max_value = keisan::clamp(value, 0, 100);}

  // Function for Contours
  void find(cv::Mat binary_mat);

  void detection(cv::Mat image) override;

  ninshiki_interfaces::msg::Contours detection_result;
  std::string color_name;

private:
  int classifier_type;

  int min_hue;
  int max_hue;
  int min_saturation;
  int max_saturation;
  int min_value;
  int max_value;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<utils::Color> colors;
};

}  // namespace ninshiki_cpp::detector

#endif  // NINSHIKI_CPP__DETECTOR__COLOR_DETECTOR_HPP_
