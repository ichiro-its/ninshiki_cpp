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

#include "ninshiki_cpp/config/grpc/call_data_load_config.hpp"

#include "ninshiki_interfaces/ninshiki.grpc.pb.h"
#include "ninshiki_interfaces/ninshiki.pb.h"
#include "rclcpp/rclcpp.hpp"
#include "nlohmann/json.hpp"

namespace ninshiki_cpp
{
CallDataLoadConfig::CallDataLoadConfig(
  ninshiki_interfaces::proto::Config::AsyncService * service, grpc::ServerCompletionQueue * cq,
  const std::string & path, std::shared_ptr<ninshiki_cpp::detector::ColorDetector> color_detection)
: CallData(service, cq, path), color_detection_(color_detection)
{
  Proceed();
}

void CallDataLoadConfig::AddNextToCompletionQueue()
{
  new CallDataLoadConfig(service_, cq_, path_, color_detection_);
}

void CallDataLoadConfig::WaitForRequest()
{
  service_->RequestLoadConfig(&ctx_, &request_, &responder_, cq_, cq_, this);
}

void CallDataLoadConfig::HandleRequest()
{
  try {
    color_detection_->load_configuration(path_);

    RCLCPP_INFO(
      rclcpp::get_logger("Load config"), "config has been loaded!"
    );
  } catch (nlohmann::json::exception e) {
    RCLCPP_ERROR(rclcpp::get_logger("Load config"), e.what());
  }
}
}  // namespace ninshiki_cpp
