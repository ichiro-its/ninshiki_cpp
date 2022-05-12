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

#include <memory>
#include <string>
#include <vector>
#include "ninshiki_cpp/node/ninshiki_cpp_node.hpp"

using namespace std::chrono_literals;

namespace ninshiki_cpp::node
{

NinshikiCppNode::NinshikiCppNode(
  rclcpp::Node::SharedPtr node, std::string topic_name, int frequency)
: node(node), dnn_detection(nullptr), color_detection(nullptr)
{
  detected_object_publisher = node->create_publisher<DetectedObjects>(
    get_node_prefix() + "/dnn_detection", 10);

  image_subscriber = node->create_subscription<Image>(
    topic_name, 10,
    [this](const Image::SharedPtr message) {
      if (!message->data.empty()) {
        // Determine whether the image is compressed or not
        if (message->quality < 0) {
          // Determine the received_frame type from the channel count
          auto type = CV_8UC1;
          if (message->channels == 2) {
            type = CV_8UC2;
          } else if (message->channels == 3) {
            type = CV_8UC3;
          } else if (message->channels == 4) {
            type = CV_8UC4;
          }
          received_frame = cv::Mat(message->rows, message->cols, type);

          // Copy the mat data from the raw image
          memcpy(received_frame.data, message->data.data(), message->data.size());
        } else {
          // Decode the compressed image
          received_frame = cv::imdecode(message->data, cv::IMREAD_UNCHANGED);
        }

        cv::cvtColor(received_frame, hsv_frame, cv::COLOR_BGR2HSV);
      }
    }
  );

  node_timer = node->create_wall_timer(
    std::chrono::milliseconds(frequency),
    [this]() {
      if (!received_frame.empty()) {
        publish();
      }
    }
  );
}

void NinshikiCppNode::publish()
{
  dnn_detection->detection(received_frame, 0.4, 0.3);
  detected_object_publisher->publish(dnn_detection->detection_result);

  color_detection->detection(hsv_frame);
  field_segmentation_publisher->publish(color_detection->detection_result);

  // Clear detection_result
  // received_frame.release();
  dnn_detection->detection_result.detected_objects.clear();
  color_detection->detection_result.contours.clear();
}

void NinshikiCppNode::set_detection(
  std::shared_ptr<DnnDetector> dnn_detection,
  std::shared_ptr<ColorDetector> color_detection)
{
  this->dnn_detection = dnn_detection;
  this->color_detection = color_detection;

  if (this->color_detection != nullptr) {
    field_segmentation_publisher = node->create_publisher<Contours>(
      get_node_prefix() + "/" + this->color_detection->name, 10);
  }
}

std::string NinshikiCppNode::get_node_prefix()
{
  return "ninshiki_cpp";
}

}  // namespace ninshiki_cpp::node
