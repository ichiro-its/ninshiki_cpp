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

#include <ninshiki_cpp/config/node/config_node.hpp>

#include <jitsuyo/config.hpp>
#include <ninshiki_cpp/utils/color.hpp>
#include <rclcpp/rclcpp.hpp>

#include <memory>
#include <string>

namespace ninshiki_cpp::detector
{

ConfigNode::ConfigNode(rclcpp::Node::SharedPtr node, const std::string & path,
  const std::shared_ptr<ColorDetector> & color_detection)
{
  get_color_setting_service = node->create_service<GetColorSetting>(
    get_node_prefix() + "/get_color_setting",
    [this, path](std::shared_ptr<GetColorSetting::Request> request,
    std::shared_ptr<GetColorSetting::Response> response) {
      nlohmann::json config;
      if (!jitsuyo::load_config(path, "color_classifier.json", config)) {
        throw std::runtime_error("Unable to open `" + path + "color_classifier.json`!");
      }
      response->json_color = config.dump();
    }
  );

  save_color_setting_service = node->create_service<SaveColorSetting>(
    get_node_prefix() + "/save_color_setting",
    [this, node, path](std::shared_ptr<SaveColorSetting::Request> request,
    std::shared_ptr<SaveColorSetting::Response> response) {
      nlohmann::json config;
      try {
        config = nlohmann::json::parse(request->json_color);
      } catch (const std::exception & e) {
        RCLCPP_ERROR(node->get_logger(), "Exception occurred: %s", e.what());
        response->success = false;
        return;
      }
      if (!jitsuyo::save_config(path, "color_classifier.json", config)) {
        response->success = false;
        return;
      }
      response->success = true;
    }
  );

  set_color_setting_service = node->create_service<SetColorSetting>(
    get_node_prefix() + "/set_color_setting",
    [this, node, path, color_detection](std::shared_ptr<SetColorSetting::Request> request,
    std::shared_ptr<SetColorSetting::Response> response) {
      try {
        utils::Color color(
          request->name,
          request->invert_hue,
          request->use_lab,
          request->min_hue, request->max_hue,
          request->min_saturation, request->max_saturation,
          request->min_value, request->max_value,
          request->min_lightness, request->max_lightness,
          request->min_a, request->max_a,
          request->min_b, request->max_b
        );

        color_detection->configure_color_setting(color);

        response->success = true;
      } catch (const std::exception & e) {
        RCLCPP_ERROR(node->get_logger(), "Exception occurred: %s", e.what());
        response->success = false;
      } catch (...) {
        RCLCPP_ERROR(node->get_logger(), "Unknown exception occurred");
        response->success = false;
      }
    }
  );
}

std::string ConfigNode::get_node_prefix() const
{
  return "ninshiki_cpp/config";
}

}  // namespace ninshiki_cpp::detector
