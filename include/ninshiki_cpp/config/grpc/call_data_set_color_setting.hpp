#ifndef NINSHIKI_CPP_CONFIG__GRPC__CALL_DATA_SET_COLOR_SETTING_HPP__
#define NINSHIKI_CPP_CONFIG__GRPC__CALL_DATA_SET_COLOR_SETTING_HPP__

#include "rclcpp/rclcpp.hpp"
#include <ninshiki_cpp/detector/color_detector.hpp>
#include <ninshiki_cpp/config/grpc/call_data.hpp>
#include "ninshiki_cpp/utils/color.hpp"

namespace ninshiki_cpp
{
class CallDataSetColorSetting
: CallData<ninshiki_interfaces::proto::ColorSetting, ninshiki_interfaces::proto::Empty>
{
public:
  CallDataSetColorSetting(
    ninshiki_interfaces::proto::Config::AsyncService * service, grpc::ServerCompletionQueue * cq,
    const std::string & path, std::shared_ptr<ninshiki_cpp::detector::ColorDetector> color_detection);

protected:
  void AddNextToCompletionQueue() override;
  void WaitForRequest();
  void HandleRequest();
  std::shared_ptr<ninshiki_cpp::detector::ColorDetector> color_detection_;
};
}  // namespace ninshiki_cpp

#endif  // NINSHIKI_CPP__CONFIG__GRPC__CALL_DATA_SET_COLOR_SETTING_HPP__