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

#include <iostream>
#include "ninshiki_cpp/node/ninshiki_cpp_node.hpp"

using namespace std::chrono_literals;

namespace ninshiki_cpp::node
{

NinshikiCppNode::NinshikiCppNode(rclcpp::Node::SharedPtr node, std::string topic_name)
: node(node), detection(nullptr)
{
  detected_object_publisher = node->create_publisher<DetectedObjects>(
    get_node_prefix() + "/detection", 10);
  
  image_subscriber = node->create_subscription<Image>(
    topic_name, 10,
    [this](const Image::SharedPtr message) {
      std::cout << "is message->data not empty" << !message->data.empty() << std::endl;

      std::cout << "quality&channels " << message->quality<<" " << message->channels << std::endl;
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
          std::cout << "type: " << type << std::endl;
          received_frame = cv::Mat(message->rows, message->cols, type);

          // Copy the mat data from the raw image
          memcpy(received_frame.data, message->data.data(), message->data.size());
        } else {
          // Decode the compressed image
          received_frame = cv::imdecode(message->data, cv::IMREAD_UNCHANGED);
        }

        // static const std::string kWinName = "Deep learning object detection in OpenCV";
        // cv::namedWindow(kWinName, cv::WINDOW_NORMAL);
        // cv::imshow(kWinName, received_frame);
        // cv::waitKey(1);
      }
      // Show From Image
      // std::string image_path = static_cast<std::string>(getenv("HOME")) + "/example_img/goalpost.jpg";
      // received_frame = cv::imread(image_path, cv::IMREAD_COLOR);
      
      

      // Save To Image
      // cv::imwrite(static_cast<std::string>(getenv("HOME")) + "/example_img/1.jpg", received_frame);
      // exit(0);

    }
  );

  node_timer = node->create_wall_timer(
    8ms,
    [this]() {

      if (!received_frame.empty()) {
        publish();
      }
    }
  );
}

void NinshikiCppNode::publish()
{
  // if (received_frame.size()) {
    detection->detection(received_frame, 0.4, 0.3);
    detected_object_publisher->publish(detection->detection_result);
  // }

  // Clear detection_result
  detection->detection_result.detected_objects.clear();
}

void NinshikiCppNode::set_detection(std::shared_ptr<ninshiki_cpp::detector::Yolo> detection)
{
  this->detection = detection;
}

std::string NinshikiCppNode::get_node_prefix()
{
  return "ninshiki_cpp";
}

}  // namespace ninshiki_cpp::node
