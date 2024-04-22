#include "ninshiki_cpp/config/grpc/call_data_save_color_setting.hpp"

#include "ninshiki_cpp/config/utils/config.hpp"
#include "ninshiki_interfaces/ninshiki.grpc.pb.h"
#include "ninshiki_interfaces/ninshiki.pb.h"
#include "rclcpp/rclcpp.hpp"
#include "nlohmann/json.hpp"

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
    // nlohmann::json color_data = nlohmann::json::parse(request_.json_color());

    std::string json_string = request_.json_color();
    // std::replace(json_string.begin(), json_string.end(), '\\', ' ');
    nlohmann::json color_data = nlohmann::json::parse(json_string);

    config.save_color_setting(color_data);
    RCLCPP_INFO(rclcpp::get_logger("Save config"), " config has been saved!  ");
  } catch (nlohmann::json::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("Save config"), e.what());
  }
}
}  // namespace ninshiki_cpp