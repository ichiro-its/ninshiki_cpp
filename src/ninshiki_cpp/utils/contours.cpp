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

#include <string>
#include <vector>
#include <limits>
#include <math.h>
#include <algorithm>

#include "ninshiki_cpp/utils/contours.hpp"
#include "opencv2/opencv.hpp"
#include "keisan/keisan.hpp"

namespace ninshiki_cpp::utils
{
  Contours::Contours()
  {
    contours.clear();
  }

  Contours::Contours(std::vector<std::vector<cv::Point>> contours)
  {
    set_contours(contours);
  }

  Contours::Contours(cv::Mat binary_mat)
  {
    find(binary_mat);
  }

  cv::Mat Contours::get_binary_mat(cv::Size mat_size)
  {
    cv::Mat binary_mat(mat_size, CV_8UC1);
    binary_mat = cv::Scalar(0);

    if (contours.size() > 0)
    {
      cv::fillPoly(binary_mat, contours, 255);
    }
    return binary_mat;
  }

  cv::Mat Contours::get_binary_mat_line(cv::Size mat_size, int line_size)
  {
    cv::Mat binary_mat(mat_size, CV_8UC1);
    binary_mat = cv::Scalar(0);

    if (contours.size() > 0)
    {
      for (unsigned int i = 0; i < contours.size(); i++)
      {
        cv::drawContours(binary_mat, contours, i, 255, line_size);
      }
    }

    return binary_mat;
  }

  std::vector<std::vector<cv::Point>> Contours::get_contours()
  {
    return contours;
  }

  void Contours::set_contours(std::vector<std::vector<cv::Point>> contours)
  {
    this->contours = contours;
  }

  void Contours::set_name(std::string contours_name)
  {
    name = contours_name;
  }

  void Contours::find(cv::Mat binary_mat)
  {
    std::vector<cv::Vec4i> hierarchy;

    contours.clear();
    cv::findContours(binary_mat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  }

  void Contours::join_all()
  {
    if (contours.size() <= 0)
      return;

    std::vector<cv::Point> join_contour;
    for (std::vector<cv::Point> & contour : contours)
    {
      for (unsigned int i = 0; i < contour.size(); i++)
      {
        join_contour.push_back(contour[i]);
      }
    }

    contours.clear();
    contours.push_back(join_contour);
  }

  void Contours::filter_smaller_than(float value)
  {
    if (contours.size() <= 0)
      return;

    std::vector<std::vector<cv::Point>> large_contours;
    for (std::vector<cv::Point> & contour : contours)
    {
      if (cv::contourArea(contour) < (double)value)
      {
        large_contours.push_back(contour);
      }
    }

    contours = large_contours;
  }

  void Contours::filter_larger_than(float value)
  {
    if (contours.size() <= 0)
      return;

    std::vector<std::vector<cv::Point>> large_contours;
    for (std::vector<cv::Point> & contour : contours)
    {
      if (cv::contourArea(contour) > (double)value)
      {
        large_contours.push_back(contour);
      }
    }

    contours = large_contours;
  }

  void Contours::filter_largest()
  {
    if (contours.size() <= 0)
      return;

    std::vector<cv::Point> largest_contour;
    double largest_contour_area = 0.0;
    for (std::vector<cv::Point> & contour : contours)
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

  std::vector<cv::Point> Contours::get_all_point_contour()
  {
    std::vector<cv::Point> all_contour;
    for (std::vector<cv::Point> & contour : contours)
    {
      for (cv::Point & point : contour)
      {
        all_contour.push_back(point);
      }
    }
    return all_contour;
  }

  float Contours::center_x()
  {
    std::vector<cv::Point> all_contour = get_all_point_contour();
    cv::Moments moments = cv::moments(all_contour, false);

    return (moments.m10 / moments.m00);
  }

  float Contours::center_y()
  {
    std::vector<cv::Point> all_contour = get_all_point_contour();
    cv::Moments moments = cv::moments(all_contour, false);

    return (moments.m01 / moments.m00);
  }

  float Contours::min_x()
  {
    float min_x = std::numeric_limits<float>::infinity(); // get infinity float
    for (std::vector<cv::Point> & contour : contours)
    {
      for (cv::Point & point : contour)
      {
        min_x = std::min<float>(min_x, point.x);
      }
    }

    return min_x;
  }

  float Contours::min_y()
  {
    float min_y = std::numeric_limits<float>::infinity(); // get infinity float
    for (std::vector<cv::Point> & contour : contours)
    {
      for (cv::Point & point : contour)
      {
        min_y = std::min<float>(min_y, point.y);
      }
    }

    return min_y;
  }

  float Contours::max_x()
  {
    float max_x = 0.0;
    for (std::vector<cv::Point> & contour : contours)
    {
      for (cv::Point & point : contour)
      {
        max_x = std::max<float>(max_x, point.x);
      }
    }

    return max_x;
  }

  float Contours::max_y()
  {
    float max_y = 0.0;
    for (std::vector<cv::Point> & contour : contours)
    {
      for (cv::Point & point : contour)
      {
        max_y = std::max<float>(max_y, point.y);
      }
    }

    return max_y;
  }

  keisan::Vector<2> Contours::min_y_point()
  {
    keisan::Vector<2> min_y_point(0.0, std::numeric_limits<float>::infinity());
    for (std::vector<cv::Point> & contour : contours)
    {
      for (cv::Point & point : contour)
      {
        if (point.y < min_y_point[1])
        {
          min_y_point[0] = point.x;
          min_y_point[1] = point.y;
        }
      }
    }

    return min_y_point;
  }

  keisan::Vector<2> Contours::max_y_point()
  {
    keisan::Vector<2> max_y_point(0.0, 0.0);
    for (std::vector<cv::Point> & contour : contours)
    {
      for (cv::Point & point : contour)
      {
        if (point.y > max_y_point[1])
        {
          max_y_point[0] = point.x;
          max_y_point[1] = point.y;
        }
      }
    }

    return max_y_point;
  }

  void Contours::expand(float value)
  {
    for (std::vector<cv::Point> & contour : contours)
    {
      if (contour.size() < 1)
        continue;

      float center_x = 0;
      float center_y = 0;

      for (cv::Point & point : contour)
      {
        center_x += point.x;
        center_y += point.y;
      }

      center_x /= contour.size();
      center_y /= contour.size();

      for (cv::Point & point : contour)
      {
        float range = std::sqrt(std::pow(point.x - center_x, 2) + std::pow(point.y - center_y, 2));

        point.x = center_x + ((point.x - center_x) * (range + value) / range);
        point.y = center_y + ((point.y - center_y) * (range + value) / range);
      }
    }
  }

  void Contours::strecth_up(float value)
  {
    for (std::vector<cv::Point> & contour : contours)
    {
      if (contour.size() < 1)
        continue;

      float center_y = 0;
      for (cv::Point & point : contour)
      {
        center_y += point.y;
      }
      center_y /= contour.size();

      for (cv::Point & point : contour)
      {
        if (point.y < center_y)
        {
          point.y = point.y - value;
        }
      }
    }
  }

  std::vector<std::vector<cv::Point>> Contours::split_left(float x)
  {
    std::vector<std::vector<cv::Point>> left_contours;

    for (std::vector<cv::Point> & contour : contours)
    {
      std::vector<cv::Point> left_contour;
      for (unsigned int i = 0; i < contour.size(); i++)
      {
        if (contour[i].x < x)
          left_contour.push_back(contour[i]);

        if (contour.size() > 1)
        {
          int x1 = contour[i].x;
          int x2 = contour[(i + 1) % contour.size()].x;
          if (x == keisan::clamp<float>(x, (x1 < x2) ? x1 : x2, (x1 > x2) ? x1 : x2))
          {
            int y1 = contour[i].y;
            int y2 = contour[(i + 1) % contour.size()].y;
            left_contour.push_back(cv::Point(x, keisan::map<float>(x, x1, x2, y1, y2)));
          }
        }
      }

      if (left_contour.size() > 2)
        left_contours.push_back(left_contour);
    }
  }
  std::vector<std::vector<cv::Point>> Contours::split_right(float x)
  {
    std::vector<std::vector<cv::Point>> right_contours;

    for (std::vector<cv::Point> & contour : contours)
    {
      std::vector<cv::Point> right_contour;
      for (unsigned int i = 0; i < contour.size(); i++)
      {
        if (contour[i].x > x)
          right_contour.push_back(contour[i]);

        if (contour.size() > 1)
        {
          int x1 = contour[i].x;
          int x2 = contour[(i + 1) % contour.size()].x;
          if (x == keisan::clamp<float>(x, (x1 < x2) ? x1 : x2, (x1 > x2) ? x1 : x2))
          {
            int y1 = contour[i].y;
            int y2 = contour[(i + 1) % contour.size()].y;
            right_contour.push_back(cv::Point(x, keisan::map<float>(x, x1, x2, y1, y2)));
          }
        }
      }

      if (right_contour.size() > 2)
        right_contours.push_back(right_contour);
    }

    return right_contours;
  }

  void Contours::filter_rect(float x, float y, float width, float height, float value)
  {
    float max_x = std::max<float>(x + width, this->max_x());
    float max_y = std::max<float>(y + height, this->max_y());

    cv::Mat binary_mat = get_binary_mat(cv::Size(max_x, max_y));
    cv::rectangle(binary_mat, cv::Rect(x, y, width, height), value, cv::FILLED);

    find(binary_mat);
  }

  void Contours::fill_rect(float x, float y, float width, float height)
  {
    filter_rect(x, y, width, height, 255);
  }
  void Contours::remove_rect(float x, float y, float width, float height)
  {
    filter_rect(x, y, width, height, 0);
  }

  void Contours::convex_hull()
  {
    if (this->contours.size() <= 0)
      return;

    std::vector<std::vector<cv::Point>> convex_hulls;
    for (std::vector<cv::Point> & contour : contours)
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

    this->contours = convex_hulls;
  }

  std::vector<std::vector<cv::Point>> Contours::left_wall_contours(float cam_width)
  {
    std::vector<std::vector<cv::Point>> left_contour;
    float max_x = 0.0;
    float min_x = INT16_MAX;
    float con_y = 0.0;
    float current_y = 0.0;

    for (std::vector<cv::Point> & contour : contours)
    {
      float current_min_x = INT16_MAX;
      for (cv::Point & point : contour)
      {
        max_x = std::max<float>(max_x, point.x);
        current_min_x = std::min<float>(current_min_x, point.x);
        current_y = std::max<float>(current_y, point.y);
      }

      if (max_x < cam_width && (current_min_x < cam_width / 4) && current_min_x < min_x && current_y > con_y)
      {
        con_y = current_y;
        min_x = current_min_x;
        left_contour.clear();
        left_contour.push_back(contour);
      }
    }

    return left_contour;
  }
  std::vector<std::vector<cv::Point>> Contours::right_wall_contours(float cam_width)
  {
    std::vector<std::vector<cv::Point>> right_contour;
    float min_x = 0.0;
    float con_y = 0.0;
    float current_y = 0.0;

    for (std::vector<cv::Point> & contour : contours)
    {
      float current_min_x = INT16_MAX;
      for (cv::Point & point : contour)
      {
        current_min_x = std::min<float>(current_min_x, point.x);
        current_y = std::max<float>(current_y, point.y);
      }

      if (current_min_x > 0.0 && current_min_x > (cam_width / 4) && current_min_x > min_x && current_y > con_y)
      {
        min_x = current_min_x;
        con_y = current_y;
        right_contour.clear();
        right_contour.push_back(contour);
      }
    }

    return right_contour;
  }

  float Contours::x_of_max_y()
  {
    float max_y = 0.0;
    float x = 0.0;

    for (std::vector<cv::Point> &contour : this->contours)
    {
      for (cv::Point & point : contour)
      {
        if (point.y > max_y)
        {
          max_y = point.y;
          x = point.x;
        }
      }
    }

    return x;
  }
} // namespace ninshiki_cpp::utils
