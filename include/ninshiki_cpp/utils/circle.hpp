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

#ifndef NINSHIKI_CPP__UTILS__CIRCLE_HPP_
#define NINSHIKI_CPP__UTILS__CIRCLE_HPP_

#include <opencv2/opencv.hpp>
#include <vector>

namespace ninshiki_cpp::utils
{

class Circle
{
private:

    cv::Point2f center;
    float radius;

public:

    Circle(const std::vector<cv::Point> & contour);

    void draw(cv::Mat & image, int line_size) const;

    const cv::Point2f & get_center() const;
    const float & get_radius() const;
};

}  // namespace ninshiki_cpp::utils

#endif // NINSHIKI_CPP__UTILS__CIRCLES_HPP_
