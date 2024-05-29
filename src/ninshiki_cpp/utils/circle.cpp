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

#include "ninshiki_cpp/utils/circle.hpp"

namespace ninshiki_cpp::utils
{

Circle::Circle(const std::vector<cv::Point> & contour)
: center(cv::Point2f(-1, -1)), radius(0.0)
{
    cv::minEnclosingCircle(contour, center, radius);
}

void Circle::draw(cv::Mat & image, int line_size) const
{
    cv::circle(image, center, radius, cv::Scalar(0, 255, 238), line_size);
}

const cv::Point2f & Circle::get_center() const
{
    return center;
}

const float & Circle::get_radius() const
{
    return radius;
}

} // namespace ninshiki_cpp::utils
