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

#include <map>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>

#include "ninshiki_cpp/detector/lbp_detector.hpp"

namespace ninshiki_cpp
{
namespace detector
{

LBPDetector::LBPDetector() {
    classifier_file_path = "../../data/" + utils::get_host_name() + "lbp_classifier/ball_cascade.xml";
    classifier_loaded = loadClassifier(classifier_file_path);
}

bool LBPDetector::loadClassifier(std::string classifier_file_path) {
    if (!cascade_detector_.load(classifier_file_path))
    {
        printf("failed to load cascade %s\n", classifier_file_path);
        return false;
    }

    return true;
}

void LBPDetector::classify(cv::Mat input)
{
    img_width = static_cast<double>(input.cols);
    img_height = static_cast<double>(input.rows);

    std::vector<cv::Rect> rects;
    if (!classifier_loaded)
        return;

    std::vector<cv::Rect> rects;
    cv::Mat gray = input.clone();
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

    cascade_detector_.detectMultiScale(gray, rects, 1.1, 5, 8, cv::Size(5, 5));

    for (auto i = rects.begin(); i != rects.end(); ++i) {
        cv::Rect rectangle = *i;
        ninshiki_interfaces::msg::DetectedObject detection_object;

        detection_object.label = "LBPDetector Detected Object";
        detection_object.left = rectangle.x / img_width;
        detection_object.top = rectangle.x / img_height;
        detection_object.right = (rectangle.x + rectangle.width) / img_width;
        detection_object.bottom = (rectangle.y + rectangle.height) / img_height;

        detection_result.detected_objects.push_back(detection_object);
    }
}

}
}
// return disamain sama color dan dnn, terus dites
// cara tesnya run ninshiki_cpp sama yg di cmake