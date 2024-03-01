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

#ifndef NINSHIKI_CPP__NODE__NINSHIKI_CPP_NODE_HPP_
#define NINSHIKI_CPP__NODE__NINSHIKI_CPP_NODE_HPP_

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "ninshiki_cpp/detector/color_detector.hpp"
#include "ninshiki_cpp/detector/detector.hpp"
#include "ninshiki_cpp/detector/dnn_detector.hpp"
#include "ninshiki_cpp/detector/lbp_detector.hpp"
#include "ninshiki_interfaces/msg/detected_objects.hpp"
#include "shisen_cpp/shisen_cpp.hpp"

namespace ninshiki_cpp::node
{

class NinshikiCppNode
{
public:
  using DnnDetector = ninshiki_cpp::detector::DnnDetector;
  using ColorDetector = ninshiki_cpp::detector::ColorDetector;
  using LBPDetector = ninshiki_cpp::detector::LBPDetector;

  NinshikiCppNode(
    rclcpp::Node::SharedPtr node,
    int frequency, shisen_cpp::Options options);
  void publish();
  void set_detection(
    std::shared_ptr<DnnDetector> dnn_detection,
    std::shared_ptr<ColorDetector> color_detection,
    std::shared_ptr<LBPDetector> lbp_detection);

private:
  using Contours = ninshiki_interfaces::msg::Contours;
  using DetectedObjects = ninshiki_interfaces::msg::DetectedObjects;
  using Image = sensor_msgs::msg::Image;

  rclcpp::Node::SharedPtr node;
  rclcpp::TimerBase::SharedPtr node_timer;

  rclcpp::Publisher<DetectedObjects>::SharedPtr detected_object_publisher;
  rclcpp::Publisher<Contours>::SharedPtr field_segmentation_publisher;
  rclcpp::Subscription<Image>::SharedPtr image_subscriber;

  std::shared_ptr<DnnDetector> dnn_detection;
  std::shared_ptr<ColorDetector> color_detection;
  std::shared_ptr<LBPDetector> lbp_detection;

  cv::Mat received_frame;
  cv::Mat hsv_frame;

  static std::string get_node_prefix();
};

}  // namespace ninshiki_cpp::node

#endif  // NINSHIKI_CPP__NODE__NINSHIKI_CPP_NODE_HPP_
