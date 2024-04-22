#ifndef NINSHIKI_CPP__CONFIG__GRPC__CONFIG_HPP_
#define NINSHIKI_CPP__CONFIG__GRPC__CONFIG_HPP_

#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_format.h>
#include <ninshiki_cpp/config/grpc/call_data.hpp>
#include <ninshiki_cpp/config/grpc/call_data_base.hpp>
#include <ninshiki_cpp/detector/color_detector.hpp>
#include <ninshiki_interfaces/ninshiki.grpc.pb.h>
#include <ninshiki_interfaces/ninshiki.pb.h>
#include <ninshiki_interfaces/msg/color_setting.hpp>
#include "grpc/support/log.h"
#include "grpcpp/grpcpp.h"
#include "nlohmann/json.hpp"
#include "rclcpp/rclcpp.hpp"

using ninshiki_interfaces::proto::Config;

namespace ninshiki_cpp
{
class ConfigGrpc
{
public:
  explicit ConfigGrpc();
  explicit ConfigGrpc(const std::string & path);
  using ColorDetector = detector::ColorDetector;

  ~ConfigGrpc();

  void Run(uint16_t port, const std::string & path, std::shared_ptr<ColorDetector> color_detection);

private:
  std::string path;
  static void SignIntHandler(int signum);

  static inline std::unique_ptr<grpc::ServerCompletionQueue> cq_;
  static inline std::unique_ptr<grpc::Server> server_;
  std::thread thread_;
  ninshiki_interfaces::proto::Config::AsyncService service_;
};

}  // namespace ninshiki_cpp

#endif  // NINSHIKI_CPP__CONFIG__GRPC__CONFIG_HPP_
