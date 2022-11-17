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

#ifndef NINSHIKI_CPP__DETECTOR__LBP_DETECTOR_HPP_
#define NINSHIKI_CPP__DETECTOR__LBP_DETECTOR_HPP_

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <vector>
#include <string>

#include "ninshiki_cpp/utils/utils.hpp"
#include "ninshiki_cpp/detector/detector.hpp"
#include "ninshiki_interfaces/msg/detected_object.hpp"
#include "ninshiki_interfaces/msg/detected_objects.hpp"

namespace ninshiki_cpp::detector
{
class LBPDetector : public Detector
{
public:
    LBPDetector();
    bool loadClassifier(std::string config_path);
    void detection(cv::Mat input);

private:
    bool classifier_loaded;
    cv::CascadeClassifier cascade_detector_;
};
}

#endif // NINSHIKI_CPP__DETECTOR__LBP_DETECTOR_HPP_
