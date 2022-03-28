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
  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (size_t i = 0; i < outs.size(); ++i)
  {
    // Network produces output blob with a shape NxC where N is a number of
    // detected objects and C is a number of classes + 4 where the first 4
    // numbers are [center_x, center_y, width, height]
    float* detection = (float*)outs[i].data;
    for (int j = 0; j < outs[i].rows; ++j, detection += outs[i].cols)
    {
      cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
      cv::Point class_id_point;
      double confidence;
      minMaxLoc(scores, 0, &confidence, 0, &class_id_point);

      if (confidence > conf_threshold)
      {
        int centerX = (int)(detection[0] * width);
        int centerY = (int)(detection[1] * height);
        int width = (int)(detection[2] * width);
        int height = (int)(detection[3] * height);
        int left = centerX - width / 2;
        int top = centerY - height / 2;

        class_ids.push_back(class_id_point.x);
        confidences.push_back((float)confidence);
        boxes.push_back(cv::Rect(left, top, width, height));
      }
    }
  }

  if (boxes.size()) {
    for (size_t i = 0; i < boxes.size(); ++i) {

      cv::Rect box = boxes[i];
      if (!box.empty()) {
        // Add detected object into vector
        ninshiki_interfaces::msg::DetectedObject detection_object;

        detection_object.label = classes[class_ids[i]];
        detection_object.score = confidences[i];
        detection_object.left = box.x / width;
        detection_object.top = box.y / height;
        detection_object.right = (box.x + box.width) / width;
        detection_object.bottom = (box.y + box.height) / height;

        detection_result.detected_objects.push_back(detection_object);
      }
    }
  }
}

}  // namespace ninshiki_cpp::detector