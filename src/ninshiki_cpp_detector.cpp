// Copyright (c) 2021-2024 ICHIRO ITS
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

#include "ninshiki_cpp/ninshiki_cpp.hpp"
#include "rclcpp/rclcpp.hpp"

#include <memory>
#include <string>
#include <vector>

int main(int argc, char ** argv)
{
  auto args = rclcpp::init_and_remove_ros_arguments(argc, argv);
  shisen_cpp::Options options;

  // Default Value
  std::string path = "";
  std::string topic_name = "";
  std::string detection_method = "yolo";
  int gpu = 0;
  int myriad = 0;
  int frequency = 96;

  const char * help_message =
    "Usage: ros2 run ninshiki_cpp detector\n"
    "[path] [--detector DETECTOR] [--GPU {0,1}] [--MYRIAD {0,1}]\n"
    "\n"
    ""
    "Positional arguments:\n"
    "path                 path to detection configuration\n"
    "\n"
    "Optional arguments:\n"
    "-h, --help           show this help message and exit\n"
    "--detector DETECTOR  chose detector we want to use (yolo / tflite)\n"
    "--GPU {0,1}          if we chose the computation using GPU\n"
    "--MYRIAD {0,1}       if we chose the computation using Compute Stick\n"
    "--frequency          specify publisher frequency";

  // Handle arguments
  try {
    if (args.size() < 2) {
      RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "Argument needed!\n\n%s", help_message);
      return 1;
    }
    int i = 1;
    int pos = 0;
    while (i < args.size()) {
      std::string arg = args[i++];
      if (arg[0] == '-') {
        if (arg == "-h" || arg == "--help") {
          RCLCPP_INFO(rclcpp::get_logger("ninshiki_cpp"), "%s", help_message);
          return 1;
        } else if (arg == "--detector") {
          if (i < args.size()) {
            std::string value = args[i++];
            if (value == "yolo" || value == "tflite") {
              detection_method = value;
            } else {
              RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "Unknown detector `%s`!\n\n%s", value.c_str(), help_message);
              return 1;
            }
          } else {
            RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "No value provided for `--detector`!\n\n%s", help_message);
            return 1;
          }
        } else if (arg == "--GPU") {
          if (i < args.size()) {
            int value = std::stoi(args[i++]);
            if (value == 0 || value == 1) {
              gpu = value;
            } else {
              RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "Unknown option for GPU `%s`!\n\n%s", std::to_string(value).c_str(), help_message);
              return 1;
            }
          } else {
            RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "No value provided for `--GPU`!\n\n%s", help_message);
            return 1;
          }
        } else if (arg == "--MYRIAD") {
          if (i < args.size()) {
            int value = std::stoi(args[i++]);
            if (value == 0 || value == 1) {
              myriad = value;
            } else {
              RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "Unknown option for MYRIAD `%s`!\n\n%s", std::to_string(value).c_str(), help_message);
              return 1;
            }
          } else {
            RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "No value provided for `--MYRIAD`!\n\n%s", help_message);
            return 1;
          }
        } else if (arg == "--frequency") {
          if (i < args.size()) {
            frequency = std::stoi(args[i++]);
          } else {
            RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "No value provided for `--frequency`!\n\n%s", help_message);
            return 1;
          }
        } else {
          RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "Unknown argument `%s`!\n\n%s", arg.c_str(), help_message);
          return 1;
        }
      } else if (pos == 0) {
        path = arg;
        ++pos;
      } else {
        RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "Unexpected positional argument `%s`!\n\n%s", arg.c_str(), help_message);
        return 1;
      }
    }
  } catch (const std::exception &e) {
    RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "Invalid arguments: %s\n\n%s", e.what(), help_message);
    return 1;
  } catch (...) {
    RCLCPP_ERROR(rclcpp::get_logger("ninshiki_cpp"), "Unknown error while parsing arguments!\n\n%s", help_message);
    return 1;
  }

  auto dnn_detection = std::make_shared<ninshiki_cpp::detector::DnnDetector>(gpu, myriad);
  auto color_detection = std::make_shared<ninshiki_cpp::detector::ColorDetector>();
  auto lbp_detection = std::make_shared<ninshiki_cpp::detector::LBPDetector>();

  color_detection->load_configuration(path);

  auto node = std::make_shared<rclcpp::Node>("ninshiki_cpp");
  auto ninshiki_cpp_node = std::make_shared<ninshiki_cpp::node::NinshikiCppNode>(
    node, path, frequency, options, dnn_detection, color_detection, lbp_detection);

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
