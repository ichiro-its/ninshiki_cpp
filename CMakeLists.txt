cmake_minimum_required(VERSION 3.5)
project(ninshiki_cpp)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(jitsuyo REQUIRED)
find_package(keisan REQUIRED)
find_package(ninshiki_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shisen_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenVINO REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "version:${OpenCV_VERSION}")
message(STATUS "libraries:${OpenCV_LIBS}")
message(STATUS "OpenCV include path:${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenVINO include path env:${OpenVINO_DIR}/../include")

add_library(${PROJECT_NAME} SHARED
  "src/${PROJECT_NAME}/config/node/config_node.cpp"
  "src/${PROJECT_NAME}/detector/color_detector.cpp"
  "src/${PROJECT_NAME}/detector/dnn_detector.cpp"
  "src/${PROJECT_NAME}/detector/lbp_detector.cpp"
  "src/${PROJECT_NAME}/node/ninshiki_cpp_node.cpp"
  "src/${PROJECT_NAME}/utils/color.cpp"
  "src/${PROJECT_NAME}/utils/contours.cpp"
  "src/${PROJECT_NAME}/utils/circle.cpp"
  "src/${PROJECT_NAME}/utils/rect.cpp"
  "src/${PROJECT_NAME}/node/ninshiki_cpp_node.cpp")

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenVINO_DIR}/../include)

ament_target_dependencies(${PROJECT_NAME}
  jitsuyo
  keisan
  ninshiki_interfaces
  OpenCV
  rclcpp
  shisen_cpp
  sensor_msgs
  cv_bridge)

target_link_libraries(${PROJECT_NAME}
  openvino::runtime
)

install(DIRECTORY "include" DESTINATION ".")

install(DIRECTORY
  launch
  DESTINATION "share/${PROJECT_NAME}")

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION "lib"
  LIBRARY DESTINATION "lib"
  RUNTIME DESTINATION "bin")

add_executable(detector "src/ninshiki_cpp_detector.cpp")
target_include_directories(detector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(detector ${PROJECT_NAME})

install(TARGETS
  detector
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(
  jitsuyo
  keisan
  ninshiki_interfaces
  OpenCV
  rclcpp
  shisen_cpp
  sensor_msgs
  cv_bridge)
ament_export_include_directories("include")
ament_export_libraries(${PROJECT_NAME})
ament_package()
