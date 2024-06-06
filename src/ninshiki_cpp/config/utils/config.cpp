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

#include <ninshiki_cpp/config/utils/config.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <string>

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
