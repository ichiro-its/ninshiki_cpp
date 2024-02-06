// Copyright (c) 2021 ICHIRO ITS
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

#include <memory>
#include <ninshiki_cpp/ninshiki_cpp.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shisen_cpp/camera/node/camera_node.hpp>
#include <shisen_cpp/camera/provider/image_provider.hpp>
#include <string>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  shisen_cpp::Options options;

  // Default Value
  std::string path = "";
  std::string topic_name = "";
  std::string detection_method = "yolo";
  int frequency = 96;

  const char * help_message =
    "Usage: ros2 run ninshiki_cpp detector\n"
    "[topic] [--detector DETECTOR]\n"
    "\n"
    "Positional arguments:\n"
    "topic                specify topic name to subscribe\n"
    "\n"
    "Optional arguments:\n"
    "-h, --help           show this help message and exit\n"
    "--detector DETECTOR  chose detector we want to use (yolo / tflite)\n"
    "--backend            chose preferable backend we want to use (default / halide / inference / "
    "opencv)\n"
    "--target             chose preferable target we want to use (yolo / tflite)\n"
    "--frequency          specify publisher frequency";

  // Handle arguments
  int backend_id = cv::dnn::DNN_BACKEND_OPENCV;
  int target_id = cv::dnn::DNN_TARGET_CPU;
  try {
    int i = 1;
    int pos = 0;
    while (i < argc) {
      std::string arg = argv[i++];
      if (arg[0] == '-') {
        if (arg == "-h" || arg == "--help") {
          std::cout << help_message << std::endl;
          return 1;
        } else if (arg == "--detector") {
          std::string value = argv[i++];
          if (value == "yolo") {
            detection_method = value;
          } else {
            std::cout << "Unknown detector `" << arg << "`!\n\n" << help_message << std::endl;
            return 1;
          }
        } else if (arg == "--frequency") {
          frequency = atoi(argv[i++]);
        } else if (arg == "--backend") {
          std::string value = argv[i++];
          if (value == "default") {
            backend_id = cv::dnn::DNN_BACKEND_DEFAULT;
          } else if (value == "halide") {
            backend_id = cv::dnn::DNN_BACKEND_HALIDE;
          } else if (value == "inference") {
            backend_id = cv::dnn::DNN_BACKEND_INFERENCE_ENGINE;
          } else if (value == "opencv") {
            backend_id = cv::dnn::DNN_BACKEND_OPENCV;
          } else {
            std::cout << "Unknown backend `" << arg << "`!\n\n" << help_message << std::endl;
            return 1;
          }
        } else if (arg == "--target") {
          std::string value = argv[i++];
          if (value == "cpu") {
            backend_id = cv::dnn::DNN_TARGET_CPU;
          } else if (value == "opencl") {
            backend_id = cv::dnn::DNN_TARGET_OPENCL;
          } else if (value == "opencl-fp16") {
            backend_id = cv::dnn::DNN_TARGET_OPENCL_FP16;
          } else if (value == "fpga") {
            backend_id = cv::dnn::DNN_TARGET_FPGA;
          } else {
            std::cout << "Unknown targets `" << arg << "`!\n\n" << help_message << std::endl;
            return 1;
          }
        } else {
          std::cout << "Unknown argument `" << arg << "`!\n\n" << help_message << std::endl;
          return 1;
        }
      } else if (pos == 0) {
        topic_name = arg;
        ++pos;
      } else if (pos == 1) {
        path = arg;
        ++pos;
      }
    }
  } catch (...) {
    std::cout << "Invalid arguments!\n\n" << help_message << std::endl;
    return 1;
  }

  auto node = std::make_shared<rclcpp::Node>("ninshiki_cpp");
  auto ninshiki_cpp_node =
    std::make_shared<ninshiki_cpp::node::NinshikiCppNode>(node, topic_name, frequency, options);

  auto dnn_detection = std::make_shared<ninshiki_cpp::detector::DnnDetector>(target_id, backend_id);
  auto color_detection = std::make_shared<ninshiki_cpp::detector::ColorDetector>();
  auto lbp_detection = std::make_shared<ninshiki_cpp::detector::LBPDetector>();

  color_detection->load_configuration(path);

  ninshiki_cpp_node->set_detection(dnn_detection, color_detection, lbp_detection);

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
