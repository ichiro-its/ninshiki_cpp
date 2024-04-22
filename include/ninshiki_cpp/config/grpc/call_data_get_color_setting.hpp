#ifndef NINSHIKI_CPP__CONFIG__GRPC__CALL_DATA_GET_COLOR_SETTING_HPP__
#define NINSHIKI_CPP__CONFIG__GRPC__CALL_DATA_GET_COLOR_SETTING_HPP__

#include <ninshiki_cpp/config/grpc/call_data.hpp>

namespace ninshiki_cpp
{
class CallDataGetColorSetting
: CallData<ninshiki_interfaces::proto::Empty, ninshiki_interfaces::proto::ConfigColor>
{
public:
  CallDataGetColorSetting(
    ninshiki_interfaces::proto::Config::AsyncService * service, grpc::ServerCompletionQueue * cq,
    const std::string & path);

protected:
  void AddNextToCompletionQueue() override;
  void WaitForRequest();
  void HandleRequest();
};
}  // namespace ninshiki_cpp

#endif  // NINSHIKI_CPP__CONFIG__GRPC__CALL_DATA_GET_COLOR_SETTING_HPP__
