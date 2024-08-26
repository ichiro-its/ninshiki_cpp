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

#include "ninshiki_cpp/detector/dnn_detector.hpp"

#include <map>
#include <string>
#include <chrono>
#include <vector>

#include "jitsuyo/linux.hpp"

namespace ninshiki_cpp
{
namespace detector
{

DnnDetector::DnnDetector()
{
  bool is_using_yolov10 = true;

  file_name = static_cast<std::string>(getenv("HOME")) + "/yolo_model/obj.names";

  if (is_using_yolov10) {
    std::string model_path = static_cast<std::string>(getenv("HOME")) + "/yolo_model/yolo_weights.onnx";
    model_suffix = jitsuyo::split_string(model_path, ".");

    model_input_shape = cv::Size(320, 320);

    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    if (model->is_dynamic()) {
      model->reshape({1, 3, static_cast<long int>(model_input_shape.height), static_cast<long int>(model_input_shape.width)});
    }

    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);

    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255, 255, 255 });
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);

    model = ppp.build();
    compiled_model = core.compile_model(model, "AUTO");
    inference_request = compiled_model.create_infer_request();

    short width, height;

    const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
    const ov::Shape input_shape = inputs[0].get_shape();

    height = input_shape[1];
    width = input_shape[2];
    model_input_shape = cv::Size2f(width, height);

    const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
    const ov::Shape output_shape = outputs[0].get_shape();

    height = output_shape[1];
    width = output_shape[2];
    model_output_shape = cv::Size(width, height);

  } else {
    std::string config = static_cast<std::string>(getenv("HOME")) + "/yolo_model/config.cfg";
    std::string model = static_cast<std::string>(getenv("HOME")) +
      "/yolo_model/yolo_weights.weights";
    net = cv::dnn::readNet(model, config, "");
    model_suffix = jitsuyo::split_string(model, ".");
  }

  std::ifstream ifs(file_name.c_str());
  std::string line;
  while (std::getline(ifs, line)) {
    classes.push_back(line);
  }


}

void DnnDetector::set_computation_method(bool gpu, bool myriad)
{
  this->gpu = gpu;
  this->myriad = myriad;

  // Set computation method (gpu, myriad, or CPU)
  if (gpu) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  } else if (myriad) {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_MYRIAD);
  } else {
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
}

void DnnDetector::detection(const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

  if (model_suffix == "weights") {
    detect_darknet(image, conf_threshold, nms_threshold);
  } else if (model_suffix == "onnx") {
    detect_onnx(image, conf_threshold, nms_threshold);
  } else {
    detect_tensorflow(image, conf_threshold, nms_threshold);
  }

  auto end_time = std::chrono::high_resolution_clock::now(); // End timing
	std::chrono::duration<double, std::milli> latency = end_time - start_time;

  sum_latency += latency.count();
  iteration_counter++;
  if (latency.count() < min_latency) {
    min_latency = latency.count();
  }

	printf("Inference time: %f ms, %d\n", latency.count(), iteration_counter);
  printf("Average latency: %f ms\n", sum_latency / iteration_counter);
  printf("-----------------------------\n");
}

void DnnDetector::detect_darknet(const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  std::vector<cv::String> layer_output = net.getUnconnectedOutLayersNames();

  // Create a 4D blob from a frame
  static cv::Mat blob;
  cv::Size input_size = cv::Size(320, 320);
  cv::dnn::blobFromImage(image, blob, 1.0, input_size, cv::Scalar(), false, false, CV_8U);

  net.setInput(blob, "", 0.00392, cv::Scalar(0, 0, 0, 0));
  std::vector<cv::Mat> outs;
  net.forward(outs, layer_output);

  // Get width and height from image
  img_width = static_cast<double>(image.cols);
  img_height = static_cast<double>(image.rows);

  static std::vector<int> out_layers = net.getUnconnectedOutLayers();
  static std::string out_layer_type = net.getLayer(out_layers[0])->type;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect2d> boxes;

  if (out_layer_type == "Region") {
    for (size_t i = 0; i < outs.size(); ++i) {
      // Network produces output blob with a shape NxC where N is a number of
      // detected objects and C is a number of classes + 4 where the first 4
      // numbers are [center_x, center_y, width, height]
      float * data = reinterpret_cast<float *>(outs[i].data);
      for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
        cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
        cv::Point class_id_point;
        double confidence;
        cv::minMaxLoc(scores, 0, &confidence, 0, &class_id_point);
        if (confidence > conf_threshold) {
          double centerX = data[0] * img_width;
          double centerY = data[1] * img_height;
          double width = data[2] * img_width;
          double height = data[3] * img_height;
          double left = centerX - width / 2;
          double top = centerY - height / 2;

          class_ids.push_back(class_id_point.x);
          confidences.push_back(static_cast<float>(confidence));
          boxes.push_back(cv::Rect2d(left, top, width, height));
        }
      }
    }
  }

  // NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends
  // we need NMS in sample or NMS is required if number of outputs > 1
  if (out_layers.size() > 1 || (out_layer_type == "Region" && (this->gpu || this->myriad))) {
    std::map<int, std::vector<size_t>> class2indices;
    for (size_t i = 0; i < class_ids.size(); i++) {
      if (confidences[i] >= conf_threshold) {
        class2indices[class_ids[i]].push_back(i);
      }
    }
    std::vector<cv::Rect2d> nms_boxes;
    std::vector<float> nms_confidences;
    std::vector<int> nms_class_ids;
    for (std::map<int, std::vector<size_t>>::iterator it = class2indices.begin();
      it != class2indices.end(); ++it)
    {
      std::vector<cv::Rect2d> local_boxes;
      std::vector<float> local_confidences;
      std::vector<size_t> class_indices = it->second;
      for (size_t i = 0; i < class_indices.size(); i++) {
        local_boxes.push_back(boxes[class_indices[i]]);
        local_confidences.push_back(confidences[class_indices[i]]);
      }
      std::vector<int> nmsIndices;
      cv::dnn::NMSBoxes(local_boxes, local_confidences, conf_threshold, nms_threshold, nmsIndices);
      for (size_t i = 0; i < nmsIndices.size(); i++) {
        size_t idx = nmsIndices[i];
        nms_boxes.push_back(local_boxes[idx]);
        nms_confidences.push_back(local_confidences[idx]);
        nms_class_ids.push_back(it->first);
      }
    }
    boxes = nms_boxes;
    class_ids = nms_class_ids;
    confidences = nms_confidences;
  }

  if (boxes.size()) {
    for (size_t i = 0; i < boxes.size(); ++i) {
      cv::Rect box = boxes[i];
      if (box.width * box.height != 0) {
        // Add detected object into vector
        ninshiki_interfaces::msg::DetectedObject detection_object;
        detection_object.label = classes[class_ids[i]];
        detection_object.score = confidences[i];
        detection_object.left = box.x;
        detection_object.top = box.y;
        detection_object.right = box.width;
        detection_object.bottom = box.height;

        detection_result.detected_objects.push_back(detection_object);
      }
    }
  }
}

void DnnDetector::detect_tensorflow(
  const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  static cv::Mat blob;
  cv::Size input_size = cv::Size(300, 300);
  cv::dnn::blobFromImage(image, blob, 1.0, input_size, cv::Scalar(), true, false, CV_8U);

  net.setInput(blob);
  cv::Mat output = net.forward();
  cv::Mat detection_mat(output.size[2], output.size[3], CV_32F, output.ptr<float>());

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect2f> boxes;

  // Get width and height from image
  img_width = static_cast<double>(image.cols);
  img_height = static_cast<double>(image.rows);

  // check detection_mat not NULL
  if (detection_mat.cols * detection_mat.rows != 0) {
    for (int i = 0; i < detection_mat.rows; i++) {
      int class_id = detection_mat.at<float>(i, 1);
      float confidence = detection_mat.at<float>(i, 2);

      // Check if the detection is of good quality
      if (confidence > conf_threshold) {
        float left = detection_mat.at<float>(i, 3) * img_width;
        float top = detection_mat.at<float>(i, 4) * img_height;
        float width = detection_mat.at<float>(i, 5) * img_width - left;
        float height = detection_mat.at<float>(i, 6) * img_height - top;

        class_ids.push_back(class_id - 1);
        confidences.push_back(confidence);
        boxes.push_back(cv::Rect2f(left, top, width, height));
      }
    }
  }

  if (boxes.size()) {
    for (size_t i = 0; i < boxes.size(); ++i) {
      cv::Rect box = boxes[i];
      if (box.width * box.height != 0) {
        // Add detected object into vector
        ninshiki_interfaces::msg::DetectedObject detection_object;
        detection_object.label = classes[class_ids[i]];
        detection_object.score = confidences[i];
        detection_object.left = box.x / img_width;
        detection_object.top = box.y / img_height;
        detection_object.right = (box.x + box.width) / img_width;
        detection_object.bottom = (box.y + box.height) / img_height;

        detection_result.detected_objects.push_back(detection_object);
      }
    }
  }
}

void DnnDetector::detect_onnx(
  const cv::Mat & image, float conf_threshold, float nms_threshold)
{
  // Preprocessing
  cv::Mat resized_image;
	cv::resize(image, resized_image, model_input_shape, 0, 0, cv::INTER_AREA);

	scale_factor.x = static_cast<float>(image.cols / model_input_shape.width);
	scale_factor.y = static_cast<float>(image.rows / model_input_shape.height);

	float *input_data = (float *)resized_image.data;
	const ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), input_data);
	inference_request.set_input_tensor(input_tensor);

  inference_request.infer();

  // Postprocessing
  const float *detections = inference_request.get_output_tensor().data<const float>();

	/*
	* 0  1  2  3      4          5
	* x, y, w. h, confidence, class_id
	*/

	for (unsigned int i = 0; i < model_output_shape.height; ++i) {
		const unsigned int index = i * model_output_shape.width;

		const float &confidence = detections[index + 4];

		if (confidence > conf_threshold) {
			const float &x = detections[index + 0];
			const float &y = detections[index + 1];
			const float &w = detections[index + 2];
			const float &h = detections[index + 3];

      // Add detected object into vector
      ninshiki_interfaces::msg::DetectedObject detection_object;

      short id = static_cast<const short>(detections[index + 5]);

      detection_object.label = classes[id];
      detection_object.score = confidence;
      detection_object.left = x * scale_factor.x;
			detection_object.top = y * scale_factor.y;
			detection_object.right = (w - x) * scale_factor.x;
			detection_object.bottom = (h - y) * scale_factor.y;

      detection_result.detected_objects.push_back(detection_object);
		}
	}
}

}  // namespace detector
}  // namespace ninshiki_cpp
