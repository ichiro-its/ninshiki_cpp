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

#ifndef NINSHIKI_CPP__UTILS__CONTOURS_HPP_
#define NINSHIKI_CPP__UTILS__CONTOURS_HPP_

#include "opencv2/opencv.hpp"
#include "keisan/keisan.hpp"

#include <string>
#include <vector>

namespace ninshiki_cpp::utils
{

class Contours
{
public:
  Contours();
  explicit Contours(std::vector<std::vector<cv::Point>> contours);
  explicit Contours(cv::Mat binary_mat);

  std::vector<std::vector<cv::Point>> contours;

  cv::Mat get_binary_mat(cv::Size mat_size);
  cv::Mat get_binary_mat_line(cv::Size mat_size, int line_size);

  void set_name(std::string contours_name);

  void find(cv::Mat binary_mat);

  void join_all();

  void filter_smaller_than(float value);
  void filter_larger_than(float value);
  void filter_largest();

  float center_x();
  float center_y();

  float min_x();
  float min_y();
  float max_x();
  float max_y();

  keisan::Vector<2> min_y_point();
  keisan::Vector<2> max_y_point();

  void expand(float value);
  void strecth_up(float value);

  std::vector<std::vector<cv::Point>> split_left(float x);
  std::vector<std::vector<cv::Point>> split_right(float x);

  void filter_rect(float x, float y, float width, float height, float value);
  void fill_rect(float x, float y, float width, float height);
  void remove_rect(float x, float y, float width, float height);

  void convex_hull();

  std::vector<std::vector<cv::Point>> left_wall_contours(float cam_width);
  std::vector<std::vector<cv::Point>> right_wall_contours(float cam_width);

  float x_of_max_y();

private:
  std::vector<cv::Point> get_all_point_contour();
  std::string name;
};

}  // namespace ninshiki_cpp::utils

#endif  // NINSHIKI_CPP__UTILS__CONTOURS_HPP_
