#include "ninshiki_cpp/config/grpc/call_data_get_color_setting.hpp"

#include "ninshiki_cpp/config/utils/config.hpp"
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
  Config config(path_);
  reply_.set_json_color(config.get_color_setting("color"));
  RCLCPP_INFO(rclcpp::get_logger("Get config"), "config has been sent!");
}
}  // namespace ninshiki_cpp
