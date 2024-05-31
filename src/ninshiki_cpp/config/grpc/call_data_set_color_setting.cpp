// Copyright (c) 2024 Ichiro ITS
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

#include <ninshiki_cpp/config/grpc/call_data_set_color_setting.hpp>
#include <ninshiki_cpp/config/utils/config.hpp>
#include <ninshiki_interfaces/ninshiki.grpc.pb.h>
#include <ninshiki_interfaces/ninshiki.pb.h>
#include <rclcpp/rclcpp.hpp>

namespace ninshiki_cpp
{
CallDataSetColorSetting::CallDataSetColorSetting(
  ninshiki_interfaces::proto::Config::AsyncService * service, grpc::ServerCompletionQueue * cq,
  const std::string & path, std::shared_ptr<ninshiki_cpp::detector::ColorDetector> color_detection)
: CallData(service, cq, path), color_detection_(color_detection)
{
  Proceed();
}

void CallDataSetColorSetting::AddNextToCompletionQueue()
{
  new CallDataSetColorSetting(service_, cq_, path_, color_detection_);
}

void CallDataSetColorSetting::WaitForRequest()
{
  service_->RequestSetColorSetting(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void CallDataSetColorSetting::HandleRequest()
{
  Config config(path_);
  try {
    utils::Color color(
      request_.name(),
      request_.min_hue(),
      request_.max_hue(),
      request_.min_saturation(),
      request_.max_saturation(),
      request_.min_value(),
      request_.max_value()
    );

    color_detection_->configure_color_setting(color);

    RCLCPP_INFO(
      rclcpp::get_logger("Set color config"), "color setting config has been applied!"
    );
  } catch (nlohmann::json::exception e) {
    RCLCPP_ERROR(rclcpp::get_logger("Set config"), e.what());
  }
}
}  // namespace ninshiki_cpp
