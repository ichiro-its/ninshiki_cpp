#include "ninshiki_cpp/config/grpc/call_data_set_color_setting.hpp"

#include "ninshiki_cpp/config/utils/config.hpp"
#include "ninshiki_interfaces/ninshiki.grpc.pb.h"
#include "ninshiki_interfaces/ninshiki.pb.h"
#include "rclcpp/rclcpp.hpp"

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
  std::cout << "[DEBUG SET DATA A]" << std::endl;
  Config config(path_);
  std::cout << "[DEBUG SET DATA B]" << std::endl;
  try {
    std::cout << "[DEBUG SET DATA C]" << std::endl;
    std::string name = request_.name();
    std::cout << "[DEBUG SET DATA D]" << std::endl;
    int min_hue = request_.min_hue();
    std::cout << "[DEBUG SET DATA E]" << std::endl;
    int max_hue = request_.max_hue();
    int min_saturation = request_.min_saturation();
    int max_saturation = request_.max_saturation();
    int min_value = request_.min_value();
    int max_value = request_.max_value();
    std::cout << "[DEBUG SET DATA F]" << std::endl;

    utils::Color color(
      name,
      min_hue,
      max_hue,
      min_saturation,
      max_saturation,
      min_value,
      max_value
    );
    std::cout << "[DEBUG SET DATA G]" << std::endl;

    color_detection_->configure_color_setting(color);
    std::cout << "[DEBUG SET DATA H]" << std::endl;

    RCLCPP_INFO(
      rclcpp::get_logger("Set color config"), "color setting config has been applied!"
    );
  } catch (nlohmann::json::exception e) {
    RCLCPP_ERROR(rclcpp::get_logger("Set config"), e.what());
  }
}
}  // namespace ninshiki_cpp