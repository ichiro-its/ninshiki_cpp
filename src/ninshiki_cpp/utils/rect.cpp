// Copyright (c) 2024 ICHIRO ITS
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

#include "ninshiki_cpp/utils/rect.hpp"

namespace ninshiki_cpp::utils
{

Rect::Rect(const cv::Rect & rect) : rect(rect) {}

const cv::Mat & Rect::get_binary_mat_line(const cv::Size & mat_size, int line_size) const
{
  cv::Mat binary_mat(mat_size, CV_8UC1, cv::Scalar(0));

  cv::rectangle(binary_mat, rect, cv::Scalar(255), line_size);

  return binary_mat;
}

const cv::Point2f & Rect::get_center() const
{
  return cv::Point2f(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

}  // namespace ninshiki_cpp::utils
