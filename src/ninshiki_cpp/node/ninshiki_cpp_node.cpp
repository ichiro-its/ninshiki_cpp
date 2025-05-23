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

#include "ninshiki_cpp/node/ninshiki_cpp_node.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace ninshiki_cpp::node
{

NinshikiCppNode::NinshikiCppNode(
  rclcpp::Node::SharedPtr node, const std::string & path,
  int frequency, std::shared_ptr<DnnDetector> dnn_detection,
  std::shared_ptr<ColorDetector> color_detection,
  std::shared_ptr<LBPDetector> lbp_detection)
: node(node), path(path), dnn_detection(dnn_detection),
  color_detection(color_detection), lbp_detection(lbp_detection)
{
  detected_object_publisher = node->create_publisher<DetectedObjects>(
    get_node_prefix() + "/dnn_detection", 10);
  color_segmentation_publisher = node->create_publisher<Contours>(
    get_node_prefix() + "/color_detection", 10);

  image_subscriber =
    node->create_subscription<Image>("camera/image", 10, [this](const Image::SharedPtr message) {
      if (!message->data.empty()) {
        received_frame = cv_bridge::toCvShare(message)->image;
      }
    });

  node_timer = node->create_wall_timer(
    std::chrono::milliseconds(frequency),
    [this]() {
      if (!received_frame.empty()) {
        cv::cvtColor(received_frame, hsv_frame, cv::COLOR_BGR2HSV);
        publish();
      }
    }
  );
}

void NinshikiCppNode::publish()
{
  if (dnn_detection) {
    dnn_detection->detection(received_frame, 0.4, 0.3);
    detected_object_publisher->publish(dnn_detection->detection_result);

    dnn_detection->detection_result.detected_objects.clear();
  }

  if (color_detection) {
    color_detection->detection(hsv_frame);
    color_segmentation_publisher->publish(color_detection->detection_result);

    color_detection->detection_result.contours.clear();
  }

  if (lbp_detection) {
    lbp_detection->detection(received_frame);
    detected_object_publisher->publish(lbp_detection->detection_result);

    lbp_detection->detection_result.detected_objects.clear();
  }

  if (!received_frame.empty()) {
    received_frame.release();
  }
}

std::string NinshikiCppNode::get_node_prefix() { return "ninshiki_cpp"; }

void NinshikiCppNode::run_config_service(const std::string & path)
{
  config_node = std::make_shared<detector::ConfigNode>(node, path, color_detection);
}

}  // namespace ninshiki_cpp::node
