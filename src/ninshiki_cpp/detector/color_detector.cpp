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
#include "ninshiki_cpp/utils/utils.hpp"

namespace ninshiki_cpp
{
namespace detector
{

std::map<int, ColorDetector *> ColorDetector::unique_instances;

ColorDetector::ColorDetector(int classifier_type)
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

  number_of_pixels = img_width * img_height;

  sync_configuration();
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

bool ColorDetector::load_configuration()
{
	std::string ss = "../../data/" + utils::get_host_name() + "/" + get_config_name();

  if (utils::is_file_exist(ss) == false)
  {
    if (save_configuration() == false)
      return false;
  }

  std::ifstream input(ss, std::ifstream::in);
  if (input.is_open() == false)
    return false;

  nlohmann::json config = nlohmann::json::parse(input);

  for (auto &[key, val] : config.items()) {
    if (key == "Color") {
      try {
        val.at("min_hue").get_to(min_hue);
        val.at("max_hue").get_to(max_hue);
        val.at("min_saturation").get_to(min_saturation);
        val.at("max_saturation").get_to(max_saturation);
        val.at("min_value").get_to(min_value);
        val.at("max_value").get_to(max_value);
      } catch (nlohmann::json::parse_error & ex) {
        std::cerr << "parse error at byte " << ex.byte << std::endl;
      }
    }
  }
  return true;
}

bool ColorDetector::save_configuration()
{
  std::string ss = "../../data/" + utils::get_host_name() + "/" + get_config_name();
  
  if (utils::is_file_exist(ss) == false)
  {
    if (utils::create_file(ss) == false)
      return false;
  }

  nlohmann::json config = {
    {"Color", {
      {"min_hue", min_hue},
      {"max_hue", max_hue},
      {"min_saturation", min_saturation},
      {"max_saturation", max_saturation},
      {"min_value", min_value},
      {"max_value", max_value}
    }}
  };

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

std::string ColorDetector::get_config_name()
{
  std::stringstream config_name;
  config_name << "color_classifier/" << name << ".json";

  return config_name.str();
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

void ColorDetector::filter_mat(uint8_t r, uint8_t g, uint8_t b)
{
  int j = -1;
  for (int i = 0; i < number_of_pixels; i++)
  {        
    if (i % img_width == 0)
      j++;
  
    if (field_binary_mat.at<uint8_t>(cv::Point(i % img_width, j)) > 0)
    {
      field_binary_mat.data[i * RGB_PIXEL_SIZE + 0] = r;
      field_binary_mat.data[i * RGB_PIXEL_SIZE + 1] = g;
      field_binary_mat.data[i * RGB_PIXEL_SIZE + 2] = b;
    }
  }
}

void ColorDetector::find()
{
  std::vector<cv::Vec4i> hierarchy;

  contours.clear();
  cv::findContours(field_binary_mat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void ColorDetector::filter_smaller_than(float value)
{
  if (contours.size() <= 0)
    return;

  std::vector<std::vector<cv::Point>> large_contours;
  for (std::vector<cv::Point> &contour : contours)
  {
    if (cv::contourArea(contour) < (double)value)
    {
      large_contours.push_back(contour);
    }
  }

  contours = large_contours;
}

void ColorDetector::filter_larger_than(float value)
{
  if (contours.size() <= 0)
    return;

  std::vector<std::vector<cv::Point>> large_contours;
  for (std::vector<cv::Point> &contour : contours)
  {
    if (cv::contourArea(contour) > (double)value)
    {
      large_contours.push_back(contour);
    }
  }

  contours = large_contours;
}

void ColorDetector::filter_largest()
{
  if (contours.size() <= 0)
    return;

  std::vector<cv::Point> largest_contour;
  double largest_contour_area = 0.0;
  for (std::vector<cv::Point> &contour : contours)
  {
    double contour_area = cv::contourArea(contour);
    if (contour_area > largest_contour_area)
    {
      largest_contour_area = contour_area;
      largest_contour = contour;
    }
  }

  if (largest_contour_area > 0.0)
  {
    contours.clear();
    contours.push_back(largest_contour);
  }
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

void ColorDetector::convex_hull()
{
  if (contours.size() <= 0)
    return;

  std::vector<std::vector<cv::Point>> convex_hulls;
  for (std::vector<cv::Point> &contour : contours)
  {
    std::vector<int> hull;
    cv::convexHull(cv::Mat(contour), hull, true);

    std::vector<cv::Point> convex_hull;
    convex_hull.push_back(contour[hull[hull.size() - 1]]);
    for (unsigned int i = 0; i < hull.size(); i++)
    {
      convex_hull.push_back(contour[hull[i]]);
    }

    convex_hulls.push_back(convex_hull);
  }

  contours = convex_hulls;
}

void ColorDetector::detection(cv::Mat image)
{
  field_binary_mat = image;

  filter_mat(0, 128, 255);
  find();
  filter_larger_than(200.0);
  join_all();
  convex_hull();
}

}  // namespace detector

}  // namespace ninshiki_cpp
