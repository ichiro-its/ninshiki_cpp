#include "ninshiki_cpp/config/utils/config.hpp"

#include <fstream>
#include <iomanip>
#include <string>

#include "nlohmann/json.hpp"

namespace ninshiki_cpp
{
Config::Config(const std::string & path) : path(path) {}

std::string Config::get_color_setting(const std::string & key) const
{
  if (key == "color") {
    std::ifstream color_file(path + "color_classifier.json");
    nlohmann::json color_data = nlohmann::json::parse(color_file);
    return color_data.dump();
  }

  return "";
}

nlohmann::json Config::get_grpc_config() const
{
  std::ifstream grpc_file(path + "grpc.json");
  nlohmann::json grpc_data = nlohmann::json::parse(grpc_file);
  grpc_file.close();
  return grpc_data;
}

void Config::save_color_setting(const nlohmann::json & color_data)
{
  std::ofstream color_file(path + "color_classifier.json", std::ios::out | std::ios::trunc);
  color_file << std::setw(2) << color_data << std::endl;
  color_file.close();
}

}  // namespace ninshiki_cpp
