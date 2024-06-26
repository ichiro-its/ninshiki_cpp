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

#include "ninshiki_cpp/config/grpc/call_data_get_color_setting.hpp"

#include "jitsuyo/config.hpp"
#include "ninshiki_interfaces/ninshiki.grpc.pb.h"
#include "ninshiki_interfaces/ninshiki.pb.h"
#include "rclcpp/rclcpp.hpp"

namespace ninshiki_cpp
{
CallDataGetColorSetting::CallDataGetColorSetting(
  ninshiki_interfaces::proto::Config::AsyncService * service, grpc::ServerCompletionQueue * cq,
  const std::string & path)
: CallData(service, cq, path)
{
  Proceed();
}

void CallDataGetColorSetting::AddNextToCompletionQueue()
{
  new CallDataGetColorSetting(service_, cq_, path_);
}

void CallDataGetColorSetting::WaitForRequest()
{
  service_->RequestGetColorSetting(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void CallDataGetColorSetting::HandleRequest()
{
  nlohmann::json data;
  if (!jitsuyo::load_config(path_, "color_classifier.json", data)) {
    RCLCPP_ERROR(rclcpp::get_logger("Get config"), "Failed to load config!");
    return;
  }
  reply_.set_json_color(data.dump());
  RCLCPP_INFO(rclcpp::get_logger("Get config"), "config has been sent!");
}
}  // namespace ninshiki_cpp
