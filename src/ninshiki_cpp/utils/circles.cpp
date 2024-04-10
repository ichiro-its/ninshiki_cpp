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

Circles::Circles()
{
    centers.clear();
    radiuses.clear();
}

cv::Mat Circles::get_binary_mat_line(cv::Size mat_size, int line_size)
{
    cv::Mat binary_mat(mat_size, CV_8UC1);
    binary_mat = cv::Scalar(0);

    if (centers.size() > 0 && radiuses.size() > 0)
    {
        for (unsigned int i = 0; i < centers.size() && i < radiuses.size(); i++)
        {
		    cv::circle(binary_mat, centers[i], radiuses[i], 255, line_size);
        }
    }

    return binary_mat;
}

cv::Point Circles::get_first_center()
{
    if (centers.size() <= 0)
        return cv::Point(-1, -1);

    return centers[0];
}

float Circles::get_first_radiuses()
{
    if (radiuses.size() <= 0)
        return 0;
    
    return radiuses[0];
}

void Circles::find(std::vector<std::vector<cv::Point>> contours)
{
    centers.clear();
    radiuses.clear();

    for (std::vector<cv::Point> &contour : contours)
    {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(cv::Mat(contour), center, radius);
        
        centers.push_back(center);
        radiuses.push_back(radius);
    }
}
} // namespace ninshiki_cpp::utils