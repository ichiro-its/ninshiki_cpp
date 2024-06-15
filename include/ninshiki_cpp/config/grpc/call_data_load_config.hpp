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

#ifndef NINSHIKI_CPP_CONFIG__GRPC__CALL_DATA_LOAD_CONFIG_HPP__
#define NINSHIKI_CPP_CONFIG__GRPC__CALL_DATA_LOAD_CONFIG_HPP__

#include "rclcpp/rclcpp.hpp"
#include "ninshiki_cpp/config/grpc/call_data.hpp"
#include "ninshiki_cpp/detector/color_detector.hpp"
#include "ninshiki_cpp/utils/color.hpp"

namespace ninshiki_cpp
{
class CallDataLoadConfig
: CallData<ninshiki_interfaces::proto::Empty, ninshiki_interfaces::proto::Empty>
{
public:
  CallDataLoadConfig(
    ninshiki_interfaces::proto::Config::AsyncService * service, grpc::ServerCompletionQueue * cq,
    const std::string & path, std::shared_ptr<ninshiki_cpp::detector::ColorDetector> color_detection);
    using ColorDetector = ninshiki_cpp::detector::ColorDetector;

protected:
  void AddNextToCompletionQueue() override;
  void WaitForRequest() override;
  void HandleRequest() override;
  std::shared_ptr<ColorDetector> color_detection_;
};
}  // namespace ninshiki_cpp

#endif  // NINSHIKI_CPP__CONFIG__GRPC__CALL_DATA_LOAD_CONFIG_HPP__
