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

#include "ninshiki_cpp/detector/yolo.hpp"

namespace ninshiki_cpp::detector
{

Yolo::Yolo(bool gpu, bool myriad)
{
  file_name = static_cast<std::string>(getenv("HOME")) + "/yolo_model/obj.names";

  std::string config = static_cast<std::string>(getenv("HOME")) + "/yolo_model/config.cfg";
  std::string weights = static_cast<std::string>(getenv("HOME")) + "/yolo_model/yolo_weights.weights";
  net = cv::dnn::readNet(config, weights);

  this->gpu = gpu;
  this->myriad = myriad;
}

void Yolo::pass_image_to_network(cv::Mat image)
{
  // Open file with classes names
  std::ifstream ifs(file_name.c_str());
  std::string line;
  while (std::getline(ifs, line))
  {
    classes.push_back(line);
  }

  // Set computation method (gpu, myriad, or CPU)
  if (gpu) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  }
  else if (myriad) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
  }
  else {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
  
  std::vector<cv::String> layer_output = net.getUnconnectedOutLayersNames();

  // Create a 4D blob from a frame
  static cv::Mat blob;
  cv::Size input_size = cv::Size(416, 416);
  cv::dnn::blobFromImage(image, blob, 1.0, input_size, cv::Scalar(), true, false, CV_8U);

  net.setInput(blob);
  std::vector<cv::Mat> outs;
  net.forward(outs, layer_output);

  // Get width and height from image
  width = image.cols;
  height = image.rows;
}

void Yolo::detection(float conf_threshold, float nms_threshold)
{

}

}  // namespace ninshiki_cpp::detector