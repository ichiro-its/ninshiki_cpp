#ifndef NINSHIKI_CPP__CONFIG__UTILS__CONFIG_HPP_
#define NINSHIKI_CPP__CONFIG__UTILS__CONFIG_HPP_

#include <fstream>
#include <map>
#include <string>

#include "nlohmann/json.hpp"

namespace ninshiki_cpp
{

class Config
{
public:
  explicit Config(const std::string & path);

  std::string get_color_setting(const std::string & key) const;
  void save_color_setting(const nlohmann::json & color_data);
  nlohmann::json get_grpc_config() const;

private:
  std::string path;
};

}  // namespace ninshiki_cpp

#endif  // NINSHIKI_CPP__CONFIG__UTILS__CONFIG_HPP_
