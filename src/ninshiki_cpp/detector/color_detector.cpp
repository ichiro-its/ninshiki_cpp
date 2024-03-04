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

#include <nlohmann/json.hpp>
#include <unistd.h>

#include <string>
#include <map>
#include <utility>
#include <vector>

#include "ninshiki_cpp/detector/color_detector.hpp"

namespace ninshiki_cpp
{
namespace detector
{

ColorDetector::ColorDetector()
{
  // sync_configuration();
  colors.clear();
}

ColorDetector::~ColorDetector()
{
}

bool ColorDetector::load_configuration(const std::string & path)
{
  config_path = path;
  std::string ss = config_path + "/color_classifier.json";

  if (utils::is_file_exist(ss) == false) {
    if (save_configuration() == false) {
      return false;
    }
  }

  std::ifstream input(ss, std::ifstream::in);
  if (input.is_open() == false) {
    return false;
  }

  nlohmann::json config = nlohmann::json::parse(input);
  for (auto & item : config.items()) {
    // Get all config
    try {
      utils::Color color(
        item.key(),
        item.value().at("min_hsv")[0],
        item.value().at("max_hsv")[0],
        item.value().at("min_hsv")[1],
        item.value().at("max_hsv")[1],
        item.value().at("min_hsv")[2],
        item.value().at("max_hsv")[2]
      );

      colors.push_back(color);
    } catch (nlohmann::json::parse_error & ex) {
      std::cerr << "parse error at byte " << ex.byte << std::endl;
    }
  }

  return true;
}

bool ColorDetector::save_configuration()
{
  std::string ss = config_path + "/color_classifier.json";

  if (utils::is_file_exist(ss) == false) {
    if (utils::create_file(ss) == false) {
      return false;
    }
  }

  nlohmann::json config = nlohmann::json::array();

  for (auto & item : colors) {
    int min_hsv[] = {item.min_hue, item.min_saturation, item.min_value};
    int max_hsv[] = {item.max_hue, item.max_saturation, item.max_value};

    nlohmann::json color = {
      {"name", item.name},
      {"min_hsv", min_hsv},
      {"max_hsv", max_hsv},
    };

    config.push_back(color);
  }

  std::ofstream output(ss, std::ofstream::out);
  if (output.is_open() == false) {
    return false;
  }

  output << config.dump(2);
  output.close();

  return true;
}

bool ColorDetector::sync_configuration()
{
  if (!load_configuration()) {
    return false;
  }

  if (!save_configuration()) {
    return false;
  }

  return true;
}

cv::Mat ColorDetector::classify(cv::Mat input)
{
  int h_min = (min_hue * 255) / 360;
  int h_max = (max_hue * 255) / 360;

  int s_min = (min_saturation * 255) / 100;
  int s_max = (max_saturation * 255) / 100;

  int v_min = (min_value * 255) / 100;
  int v_max = (max_value * 255) / 100;

  cv::Scalar hsv_min = cv::Scalar(h_min, s_min, v_min);
  cv::Scalar hsv_max = cv::Scalar(h_max, s_max, v_max);

  cv::Mat output = input.clone();

  cv::inRange(input, hsv_min, hsv_max, output);

  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
  cv::morphologyEx(output, output, cv::MORPH_CLOSE, element);

  return output;
}

cv::Mat ColorDetector::classify_gray(cv::Mat input)
{
  int v_min = (min_value * 255) / 100;
  int v_max = (max_value * 255) / 100;

  cv::Mat output = input.clone();

  int pixel_num = input.cols * input.rows;
  int j = -1;
  for (int i = 0; i < pixel_num; i++) {
    if (i % input.cols == 0) {
      j++;
    }

    cv::Point pixel(i % input.cols, j);
    output.at<uint8_t>(pixel) = (input.at<uint8_t>(pixel) > v_min &&
      input.at<uint8_t>(pixel) < v_max);
  }

  return output;
}

void ColorDetector::find(cv::Mat binary_mat)
{
  std::vector<cv::Vec4i> hierarchy;

  contours.clear();
  cv::findContours(
    binary_mat, contours, hierarchy, cv::RETR_LIST,
    cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void ColorDetector::detection(const cv::Mat & image)
{
  // Get width and height from image
  double img_width = static_cast<double>(image.cols);
  double img_height = static_cast<double>(image.rows);

  // iterate every color in colors
  for (auto & color : colors) {
    color_name = color.name;
    min_hue = color.min_hue;
    max_hue = color.max_hue;
    min_saturation = color.min_saturation;
    max_saturation = color.max_saturation;
    min_value = color.min_value;
    max_value = color.max_value;

    cv::Mat field_binary_mat = classify(image);
    find(field_binary_mat);

    // Copy contours to ros2 msg
    if (contours.size() >= 0) {
      for (std::vector<cv::Point> & contour : contours) {
        ninshiki_interfaces::msg::Contour contour_msg;

        for (cv::Point & point : contour) {
          ninshiki_interfaces::msg::Point point_msg;
          point_msg.x = static_cast<float>(point.x) / img_width;
          point_msg.y = static_cast<float>(point.y) / img_height;

          contour_msg.name = color_name;
          contour_msg.contour.push_back(point_msg);
        }
        detection_result.contours.push_back(contour_msg);
      }
    }
  }
}
}  // namespace detector
}  // namespace ninshiki_cpp
