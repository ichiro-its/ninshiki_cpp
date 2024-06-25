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
find_package(keisan REQUIRED)
find_package(ninshiki_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(shisen_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")

add_library(${PROJECT_NAME} SHARED
  "src/${PROJECT_NAME}/config/grpc/config.cpp"
  "src/${PROJECT_NAME}/config/grpc/call_data_get_color_setting.cpp"
  "src/${PROJECT_NAME}/config/grpc/call_data_load_config.cpp"
  "src/${PROJECT_NAME}/config/grpc/call_data_save_color_setting.cpp"
  "src/${PROJECT_NAME}/config/grpc/call_data_set_color_setting.cpp"
  "src/${PROJECT_NAME}/config/utils/config.cpp"
  "src/${PROJECT_NAME}/detector/color_detector.cpp"
  "src/${PROJECT_NAME}/detector/dnn_detector.cpp"
  "src/${PROJECT_NAME}/detector/lbp_detector.cpp"
  "src/${PROJECT_NAME}/node/ninshiki_cpp_node.cpp"
  "src/${PROJECT_NAME}/utils/color.cpp"
  "src/${PROJECT_NAME}/utils/contours.cpp"
  "src/${PROJECT_NAME}/utils/circle.cpp"
  "src/${PROJECT_NAME}/node/ninshiki_cpp_node.cpp"
  "src/${PROJECT_NAME}/utils/utils.cpp")

add_library(${PROJECT_NAME}_exported SHARED
  "src/${PROJECT_NAME}/detector/color_detector.cpp"
  "src/${PROJECT_NAME}/detector/dnn_detector.cpp"
  "src/${PROJECT_NAME}/detector/lbp_detector.cpp"
  "src/${PROJECT_NAME}/utils/color.cpp"
  "src/${PROJECT_NAME}/utils/contours.cpp"
  "src/${PROJECT_NAME}/utils/circle.cpp"
  "src/${PROJECT_NAME}/utils/utils.cpp")

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_include_directories(${PROJECT_NAME}_exported PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

ament_target_dependencies(${PROJECT_NAME}
  keisan
  ninshiki_interfaces
  OpenCV
  rclcpp
  shisen_cpp
  sensor_msgs
  cv_bridge
  gRPC)

ament_target_dependencies(${PROJECT_NAME}_exported
  keisan
  ninshiki_interfaces
  OpenCV
  rclcpp
  shisen_cpp
  sensor_msgs
  cv_bridge)

target_link_libraries(${PROJECT_NAME}
  gRPC::grpc++_reflection
  gRPC::grpc++
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

install(TARGETS ${PROJECT_NAME}_exported
  EXPORT export_${PROJECT_NAME}_exported
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
  keisan
  ninshiki_interfaces
  OpenCV
  rclcpp
  shisen_cpp
  sensor_msgs
  cv_bridge)
ament_export_include_directories("include")
ament_export_libraries(${PROJECT_NAME}_exported)
ament_package()
