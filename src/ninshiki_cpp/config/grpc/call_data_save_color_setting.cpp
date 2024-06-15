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

#include "ninshiki_cpp/config/grpc/call_data_save_color_setting.hpp"
#include "ninshiki_cpp/config/utils/config.hpp"
#include "ninshiki_interfaces/ninshiki.grpc.pb.h"
#include "ninshiki_interfaces/ninshiki.pb.h"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"

namespace ninshiki_cpp
{
CallDataSaveColorSetting::CallDataSaveColorSetting(
  ninshiki_interfaces::proto::Config::AsyncService * service, grpc::ServerCompletionQueue * cq,
  const std::string & path)
: CallData(service, cq, path)
{
  Proceed();
}

void CallDataSaveColorSetting::AddNextToCompletionQueue()
{
  new CallDataSaveColorSetting(service_, cq_, path_);
}

void CallDataSaveColorSetting::WaitForRequest()
{
  service_->RequestSaveColorSetting(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void CallDataSaveColorSetting::HandleRequest()
{
  Config config(path_);
  try {
    std::string json_string = request_.json_color();
    nlohmann::json color_data = nlohmann::json::parse(json_string);

    config.save_color_setting(color_data);
    RCLCPP_INFO(rclcpp::get_logger("Save config"), "config has been saved!");
  } catch (nlohmann::json::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("Save config"), e.what());
  }
}
}  // namespace ninshiki_cpp
