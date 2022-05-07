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

#include "ninshiki_cpp/detector/color_detector.hpp"

namespace ninshiki_cpp
{
namespace detector
{

std::map<int, ColorDetector *> ColorDetector::unique_instances;

ColorDetector::ColorDetector(int classifier_type)
: config_path("")
{
  this->classifier_type = classifier_type;
  unique_instances.insert(std::pair<int, ColorDetector *>(this->classifier_type, this));

  switch (this->classifier_type)
  {
    case CLASSIFIER_TYPE_RED: name = "red"; break;
    case CLASSIFIER_TYPE_BLUE: name = "blue"; break;
    case CLASSIFIER_TYPE_YELLOW: name = "yellow"; break;
    case CLASSIFIER_TYPE_CYAN: name = "cyan"; break;
    case CLASSIFIER_TYPE_MAGENTA: name = "magenta"; break;
    case CLASSIFIER_TYPE_BALL: name = "ball"; break;
    case CLASSIFIER_TYPE_FIELD: name = "field"; break;
    case CLASSIFIER_TYPE_GOAL: name = "goal"; break;
    case CLASSIFIER_TYPE_MARATHON: name = "marathon"; break;
    case CLASSIFIER_TYPE_BLACK: name = "black"; break;
    case CLASSIFIER_TYPE_WHITE: name = "white"; break;
    case CLASSIFIER_TYPE_BASKETBALL: name = "basketball"; break;
    case CLASSIFIER_TYPE_LINE: name = "line"; break;
  }

  // sync_configuration();
  colors.clear();
}

ColorDetector::~ColorDetector()
{
  unique_instances.erase(classifier_type);
}

ColorDetector *ColorDetector::get_instance(int classifier_type)
{
  if (unique_instances.find(classifier_type) != unique_instances.end())
    return unique_instances[classifier_type];

  return (new ColorDetector(classifier_type));
}

ColorDetector *ColorDetector::get_instance(std::string name)
{
  for (const std::pair<int, ColorDetector*> &keyval: unique_instances)
  {
    if (keyval.second->name == name)
      return keyval.second;
  }

  return nullptr;
}

bool ColorDetector::load_configuration(const std::string & path)
{
  config_path = path;
	std::string ss = config_path + "/color_classifier.json";

  if (utils::is_file_exist(ss) == false)
  {
    if (save_configuration() == false)
      return false;
  }

  std::ifstream input(ss, std::ifstream::in);
  if (input.is_open() == false)
    return false;

  nlohmann::json config = nlohmann::json::parse(input);
  for (auto &item : config.items()) {
    // Get all config
    try {
      utils::Color color(
        item.value().at("name"),
        item.value().at("min_hsv")[0],
        item.value().at("min_hsv")[1],
        item.value().at("min_hsv")[2],
        item.value().at("max_hsv")[0],
        item.value().at("max_hsv")[1],
        item.value().at("max_hsv")[2]
      );

      // Assign hsv value based on name
      if (item.value().at("name") == name) {
        min_hue = color.min_hue;
        max_hue = color.max_hue;
        min_saturation = color.min_saturation;
        max_saturation = color.max_saturation;
        min_value = color.min_value;
        max_value = color.max_value;
      }
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
  
  if (utils::is_file_exist(ss) == false)
  {
    if (utils::create_file(ss) == false)
      return false;
  }

  nlohmann::json config = nlohmann::json::array();

  for (auto &item : colors) {
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
  if (output.is_open() == false)
    return false;

  output << config.dump(2);
  output.close();

  return true;
}

bool ColorDetector::sync_configuration()
{
  if (!load_configuration())
    return false;

  if (!save_configuration())
    return false;

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
  int v_min = (min_value* 255) / 100;
  int v_max = (max_value* 255) / 100;

  cv::Mat output = input.clone();

  int pixel_num = input.cols * input.rows;
  int j = -1;
  for (int i = 0; i < pixel_num; i++)
  {
    if (i % input.cols == 0)
      j++;

    cv::Point pixel(i % input.cols, j);
    output.at<uint8_t>(pixel) = (input.at<uint8_t>(pixel) > v_min && input.at<uint8_t>(pixel) < v_max);
  }

  return output;
}

void ColorDetector::find(cv::Mat binary_mat)
{
  std::vector<cv::Vec4i> hierarchy;

  contours.clear();
  cv::findContours(binary_mat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void ColorDetector::join_all()
{
  if (contours.size() <= 0)
    return;

  std::vector<cv::Point> join_contour;
  for (std::vector<cv::Point> &contour : contours)
  {
    for (unsigned int i = 0; i < contour.size(); i++)
    {
      join_contour.push_back(contour[i]);
    }
  }

  contours.clear();
  contours.push_back(join_contour);
}

void ColorDetector::detection(cv::Mat image)
{
  cv::Mat field_binary_mat = classify(image);
  find(field_binary_mat);
  // join_all();

  // Copy contours to ros2 msg
  if (contours.size() >= 0) {
    for (std::vector<cv::Point> &contour : contours)
    {
      ninshiki_interfaces::msg::Contour contour_msg;

      for (cv::Point &point : contour)
      {
        ninshiki_interfaces::msg::Point point_msg;
        point_msg.x = point.x;
        point_msg.y = point.y;

        contour_msg.contour.push_back(point_msg);
      }
      detection_result.contours.push_back(contour_msg);
    }
  }
}

}  // namespace detector

}  // namespace ninshiki_cpp
