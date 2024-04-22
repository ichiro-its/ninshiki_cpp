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
  rclcpp::Node::SharedPtr node, std::string topic_name,
  int frequency, shisen_cpp::Options options, std::string path)
: node(node), dnn_detection(nullptr), color_detection(nullptr), path(path)
{
  std::cout << "[DEBUG NODE A]" << std::endl;
  detected_object_publisher = node->create_publisher<DetectedObjects>(
    get_node_prefix() + "/dnn_detection", 10);
  std::cout << "[DEBUG NODE B]" << std::endl;
  field_segmentation_publisher = node->create_publisher<Contours>(
    get_node_prefix() + "/color_detection", 10);
  std::cout << "[DEBUG NODE C]" << std::endl;

  image_provider = std::make_shared<shisen_cpp::camera::ImageProvider>(options);
  std::cout << "[DEBUG NODE D]" << std::endl;

  node_timer = node->create_wall_timer(
    std::chrono::milliseconds(frequency),
    [this]() {
      // std::cout << "[DEBUG NODE TIMER]" << std::endl;
      image_provider->update_mat();
      received_frame = image_provider->get_mat();
      if (!received_frame.empty()) {
        publish();
      }
    }
  );
  std::cout << "[DEBUG NODE E]" << std::endl;
  std::cout << "path: " << path << std::endl;
  config_grpc.Run(5757, path, color_detection);
  RCLCPP_INFO(rclcpp::get_logger("GrpcServers"), "grpc running");

  std::cout << "[DEBUG NODE F]" << std::endl;
}

void NinshikiCppNode::publish()
{
  // dnn_detection->detection(received_frame, 0.4, 0.3);
  // detected_object_publisher->publish(dnn_detection->detection_result);
  // std::cout << "[DEBUG NODE PUBLISH A]" << std::endl;
  // color_detection->detection(hsv_frame);
  color_detection->detection(received_frame);
  // std::cout << "[DEBUG NODE PUBLISH B]" << std::endl;
  field_segmentation_publisher->publish(color_detection->detection_result);
  // std::cout << "[DEBUG NODE PUBLISH C]" << std::endl;

  // lbp_detection->detection(received_frame);
  // detected_object_publisher->publish(lbp_detection->detection_result);

  // Clear detection_result
  // received_frame.release();
  // dnn_detection->detection_result.detected_objects.clear();
  color_detection->detection_result.contours.clear();
  // lbp_detection->detection_result.detected_objects.clear();
  // std::cout << "[DEBUG NODE PUBLISH D]" << std::endl;
}

void NinshikiCppNode::set_detection(
  std::shared_ptr<DnnDetector> dnn_detection,
  std::shared_ptr<ColorDetector> color_detection,
  std::shared_ptr<LBPDetector> lbp_detection)
{
  // this->dnn_detection = dnn_detection;
  this->color_detection = color_detection;
  // this->lbp_detection = lbp_detection;
}

std::string NinshikiCppNode::get_node_prefix()
{
  return "ninshiki_cpp";
}

}  // namespace ninshiki_cpp::node
