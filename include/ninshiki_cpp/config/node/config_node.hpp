// Copyright (c) 2025 Ichiro ITS
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

#ifndef NINSHIKI_CPP__CONFIG__NODE__CONFIG_NODE_HPP_
#define NINSHIKI_CPP__CONFIG__NODE__CONFIG_NODE_HPP_

#include <ninshiki_interfaces/srv/get_color_setting.hpp>
#include <ninshiki_interfaces/srv/save_color_setting.hpp>
#include <ninshiki_interfaces/srv/set_color_setting.hpp>
#include <ninshiki_cpp/detector/color_detector.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>

namespace ninshiki_cpp::detector
{

class ConfigNode
{
public:
  using GetColorSetting = ninshiki_interfaces::srv::GetColorSetting;
  using SaveColorSetting = ninshiki_interfaces::srv::SaveColorSetting;
  using SetColorSetting = ninshiki_interfaces::srv::SetColorSetting;

  explicit ConfigNode(rclcpp::Node::SharedPtr node, const std::string & path,
    const std::shared_ptr<ColorDetector> & color_detection);

private:
  std::string get_node_prefix() const;

  rclcpp::Service<GetColorSetting>::SharedPtr get_color_setting_service;
  rclcpp::Service<SaveColorSetting>::SharedPtr save_color_setting_service;
  rclcpp::Service<SetColorSetting>::SharedPtr set_color_setting_service;
};

}  // namespace ninshiki_cpp::detector

#endif  // NINSHIKI_CPP__CONFIG__NODE__CONFIG_NODE_HPP_
