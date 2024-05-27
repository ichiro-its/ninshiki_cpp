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

#include "ninshiki_cpp/utils/circles.hpp"

namespace ninshiki_cpp::utils
{

Circles::Circles(const std::vector<std::vector<cv::Point>> & contours)
: centers{}, radiuses{}
{
    centers.resize(contours.size());
    radiuses.resize(contours.size());

    for (int i = contours.size() - 1; i >= 0; --i)
    {
        cv::minEnclosingCircle(cv::Mat(contours[i]), centers[i], radiuses[i]);
    }
}

cv::Mat Circles::get_binary_mat_line(const cv::Size & mat_size, int line_size)
{
    cv::Mat binary_mat(mat_size, CV_8UC1);
    binary_mat = cv::Scalar(0);

    if (centers.size() > 0 && radiuses.size() > 0)
    {
        for (int i = std::min(centers.size(), radiuses.size()) - 1; i >= 0; --i)  
        {  
            cv::circle(binary_mat, centers[i], radiuses[i], 255, line_size);  
        } 
    }

    return binary_mat;
}

cv::Point2f Circles::get_first_center()
{
    if (centers.empty())
        return cv::Point2f(-1, -1);

    return centers[0];
}

float Circles::get_first_radiuses()
{
    if (radiuses.empty())
        return 0;
    
    return radiuses[0];
}

} // namespace ninshiki_cpp::utils
