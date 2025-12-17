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

#include "jitsuyo/config.hpp"
#include "jitsuyo/linux.hpp"
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

  if (jitsuyo::is_file_exist(ss) == false) {
    if (save_configuration() == false) {
      return false;
    }
  }

  nlohmann::json config;
  if (!jitsuyo::load_config(path, "/color_classifier.json", config)) {
    return false;
  }

  for (auto & item : config.items()) {
    // Get all config
    try {
      utils::Color color(
        item.key(),
        utils::Color::Config{
          .invert_hue = item.value().at("invert_hue").get<bool>(),
          .use_lab = item.value().at("use_lab").get<bool>(),
          .min_hue = item.value().at("min_hsv")[0],
          .max_hue = item.value().at("max_hsv")[0],
          .min_saturation = item.value().at("min_hsv")[1],
          .max_saturation = item.value().at("max_hsv")[1],
          .min_value = item.value().at("min_hsv")[2],
          .max_value = item.value().at("max_hsv")[2],
          .min_lightness = item.value().at("min_lab")[0],
          .max_lightness = item.value().at("max_lab")[0],
          .min_a = item.value().at("min_lab")[1],
          .max_a = item.value().at("max_lab")[1],
          .min_b = item.value().at("min_lab")[2],
          .max_b = item.value().at("max_lab")[2]
        }
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

  if (jitsuyo::is_file_exist(ss) == false) {
    if (jitsuyo::create_file(ss) == false) {
      return false;
    }
  }

  nlohmann::json config = nlohmann::json::array();

  for (auto & item : colors) {
    bool invert_hue = item.config.invert_hue;
    bool use_lab = item.config.use_lab;
    int min_hsv[] = {item.config.min_hue, item.config.min_saturation, item.config.min_value};
    int max_hsv[] = {item.config.max_hue, item.config.max_saturation, item.config.max_value};
    int min_lab[] = {item.config.min_lightness, item.config.min_a, item.config.min_b};
    int max_lab[] = {item.config.max_lightness, item.config.max_a, item.config.max_b};

    nlohmann::json color = {
      {item.name, {
        {"invert_hue", invert_hue},
        {"use_lab", use_lab},
        {"min_hsv", min_hsv},
        {"max_hsv", max_hsv},
        {"min_lab", min_lab},
        {"max_lab", max_lab},
      }}
    };

    config.push_back(color);
  }

  return jitsuyo::save_config(config_path, "/color_classifier.json", config);
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

void ColorDetector::configure_color_setting(utils::Color color)
{
  for (auto & item : colors) {
    if (item.name == color.name) {
      item.config = color.config;

      break;
    }
  }
}

cv::Mat ColorDetector::classify(cv::Mat input)
{
  int h_min = (min_hue * 180) / 360;
  int h_max = (max_hue * 180) / 360;

  int s_min = (min_saturation * 255) / 100;
  int s_max = (max_saturation * 255) / 100;

  int v_min = (min_value * 255) / 100;
  int v_max = (max_value * 255) / 100;

  cv::Scalar hsv_min = cv::Scalar(h_min, s_min, v_min);
  cv::Scalar hsv_max = cv::Scalar(h_max, s_max, v_max);

  cv::Mat output = input.clone();

  if (invert_hue) {
    cv::Mat mask1, mask2;

    cv::inRange(input,
      cv::Scalar(0, s_min, v_min),
      cv::Scalar(h_min, s_max, v_max),
      mask1
    );

    cv::inRange(input,
      cv::Scalar(h_max, s_min, v_min),
      cv::Scalar(180, s_max, v_max),
      mask2
    );

    cv::bitwise_or(mask1, mask2, output);
  } else {
    cv::inRange(input, hsv_min, hsv_max, output);
  }

  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
  cv::morphologyEx(output, output, cv::MORPH_CLOSE, element);

  return output;
}

cv::Mat ColorDetector::classify_lab(cv::Mat input)
{
  int l_min = min_lightness;
  int l_max = max_lightness;

  int a_min = min_a + 128;
  int a_max = max_a + 128;

  int b_min = min_b + 128;
  int b_max = max_b + 128;

  cv::Scalar lab_min = cv::Scalar(l_min, a_min, b_min);
  cv::Scalar lab_max = cv::Scalar(l_max, a_max, b_max);

  cv::Mat output = input.clone();

  cv::inRange(input, lab_min, lab_max, output);

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
  // iterate every color in colors
  for (auto & color : colors) {
    color_name = color.name;
    invert_hue = color.config.invert_hue;
    use_lab = color.config.use_lab;
    min_hue = color.config.min_hue;
    max_hue = color.config.max_hue;
    min_saturation = color.config.min_saturation;
    max_saturation = color.config.max_saturation;
    min_value = color.config.min_value;
    max_value = color.config.max_value;
    min_lightness = color.config.min_lightness;
    max_lightness = color.config.max_lightness;
    min_a = color.config.min_a;
    max_a = color.config.max_a;
    min_b = color.config.min_b;
    max_b = color.config.max_b;

    if (!use_lab) {
      cv::Mat hsv_image;
      cv::cvtColor(image, hsv_image, cv::COLOR_BGR2HSV);
      cv::Mat field_binary_mat = classify(hsv_image);
      find(field_binary_mat);
    } else {
      cv::Mat lab_image;
      cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);
      cv::Mat field_binary_mat = classify_lab(lab_image);
      find(field_binary_mat);
    }

    // Copy contours to ros2 msg
    if (contours.size() >= 0) {
      for (std::vector<cv::Point> & contour : contours) {
        ninshiki_interfaces::msg::Contour contour_msg;

        for (cv::Point & point : contour) {
          ninshiki_interfaces::msg::Point point_msg;
          point_msg.x = static_cast<float>(point.x);
          point_msg.y = static_cast<float>(point.y);

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
