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

#include <iostream>
#include "ninshiki_cpp/detector/yolo.hpp"

namespace ninshiki_cpp::detector
{

Yolo::Yolo(bool gpu, bool myriad)
{
  file_name = static_cast<std::string>(getenv("HOME")) + "/yolo_model/obj.names";
  std::string config = static_cast<std::string>(getenv("HOME")) + "/yolo_model/config.cfg";
  std::string weights = static_cast<std::string>(getenv("HOME")) + "/yolo_model/yolo_weights.weights";
  net = std::make_shared<cv::dnn::Net>(cv::dnn::readNet(weights, config, ""));

  this->gpu = gpu;
  this->myriad = myriad;
}

void Yolo::detection(const cv::Mat& image, float conf_threshold, float nms_threshold)
{
  std::ifstream ifs(file_name.c_str());
  std::string line;
  while (std::getline(ifs, line))
  {
    classes.push_back(line);
  }

  // Set computation method (gpu, myriad, or CPU)
  if (gpu) {
    net->setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net->setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  }
  else if (myriad) {
    net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net->setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
  }
  else {
    net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
  
  std::vector<cv::String> layer_output = net->getUnconnectedOutLayersNames();

  // Create a 4D blob from a frame
  static cv::Mat blob;
  cv::Size input_size = cv::Size(416, 416);
  cv::dnn::blobFromImage(image, blob, 1.0, input_size, cv::Scalar(), true, false, CV_8U);

  net->setInput(blob, "", 0.00392, cv::Scalar(0,0,0,0));
  std::vector<cv::Mat> outs;
  net->forward(outs, layer_output);

  // Get width and height from image
  width = image.cols;
  height = image.rows;

  static std::vector<int> out_layers = net->getUnconnectedOutLayers();
  static std::string out_layer_type = net->getLayer(out_layers[0])->type;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  std::cout << "outs.size(): " << outs.size() << std::endl;

  for (size_t i = 0; i < outs.size(); ++i)
  {
    // Network produces output blob with a shape NxC where N is a number of
    // detected objects and C is a number of classes + 4 where the first 4
    // numbers are [center_x, center_y, width, height]
    std::cout << "outs: " << (float*)outs[i].data << std::endl;

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

  // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
  // or NMS is required if number of outputs > 1
  if (out_layers.size() > 1 || out_layer_type == "Region")
  {
    std::map<int, std::vector<size_t> > class2indices;
    for (size_t i = 0; i < class_ids.size(); i++)
    {
        if (confidences[i] >= conf_threshold)
        {
            class2indices[class_ids[i]].push_back(i);
        }
    }
    std::vector<cv::Rect> nmsBoxes;
    std::vector<float> nmsConfidences;
    std::vector<int> nmsClassIds;
    for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
    {
        std::vector<cv::Rect> localBoxes;
        std::vector<float> localConfidences;
        std::vector<size_t> classIndices = it->second;
        for (size_t i = 0; i < classIndices.size(); i++)
        {
            localBoxes.push_back(boxes[classIndices[i]]);
            localConfidences.push_back(confidences[classIndices[i]]);
        }
        std::vector<int> nmsIndices;
        cv::dnn::NMSBoxes(localBoxes, localConfidences, conf_threshold, nms_threshold, nmsIndices);
        for (size_t i = 0; i < nmsIndices.size(); i++)
        {
            size_t idx = nmsIndices[i];
            nmsBoxes.push_back(localBoxes[idx]);
            nmsConfidences.push_back(localConfidences[idx]);
            nmsClassIds.push_back(it->first);
        }
    }
    boxes = nmsBoxes;
    class_ids = nmsClassIds;
    confidences = nmsConfidences;
  }

  std::cout << "boxes.size(): " << boxes.size() << std::endl;
  if (boxes.size()) {
    for (size_t i = 0; i < boxes.size(); ++i) {

      cv::Rect box = boxes[i];
      std::cout << "box: " << box << std::endl;
      std::cout << "--------box area " << box.width << " " << box.height << std::endl;
      if (box.empty()) {
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
  //std::cout << "detection_result.detected_objects: " << detection_result.detected_objects << std::endl;

}

}  // namespace ninshiki_cpp::detector