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
#include <string>
#include "rclcpp/rclcpp.hpp"

#include "ninshiki_cpp/detector/yolo.hpp"
#include "ninshiki_interfaces/msg/detected_objects.hpp"
#include "shisen_interfaces/msg/image.hpp"

namespace ninshiki_cpp::node
{

class NinshikiCppNode
{
public:
  NinshikiCppNode(rclcpp::Node::SharedPtr node, std::string topic_name);
  void publish();
  void set_detection(std::shared_ptr<ninshiki_cpp::detector::Yolo> detection);


private:
  using DetectedObjects = ninshiki_interfaces::msg::DetectedObjects;
  using Image = shisen_interfaces::msg::Image;

  rclcpp::Node::SharedPtr node;
  rclcpp::TimerBase::SharedPtr node_timer;

  rclcpp::Subscription<shisen_interfaces::msg::Image>::SharedPtr image_subscriber;
  rclcpp::Publisher<DetectedObjects>::SharedPtr detected_object_publisher;

  std::shared_ptr<ninshiki_cpp::detector::Yolo> detection;
  
  cv::Mat received_frame;

  static std::string get_node_prefix();
};

}  // namespace ninshiki_cpp::node

#endif  // NINSHIKI_CPP__NODE__NINSHIKI_CPP_NODE_HPP_