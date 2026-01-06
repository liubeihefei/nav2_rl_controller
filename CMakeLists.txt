cmake_minimum_required(VERSION 3.8)
project(nav2_rl_controller)

# 注意：确保使用系统默认 GCC（gcc-11）编译，以匹配 ONNX Runtime 1.23.2 的 ABI
# 避免使用 gcc-toolset-14，因为它会导致 C++ ABI 不兼容
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-std=c++17 -Wall -Wextra -Wpedantic)
endif()

# 指定 ONNX Runtime 路径
set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBRARIES "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(tf2_ros REQUIRED)

# ONNX Runtime
# find_package(onnxruntime REQUIRED)

include_directories(include)

add_library(${PROJECT_NAME} SHARED
  src/rl_controller.cpp
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  geometry_msgs
  nav_msgs
  std_msgs
  pluginlib
  tf2_ros
)

# Link to ONNX Runtime
target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIBRARIES})

pluginlib_export_plugin_description_file(nav2_core nav2_rl_controller.xml)

install(
  DIRECTORY include/
  DESTINATION include
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES nav2_rl_controller.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY params
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  # find_package(ament_lint_auto REQUIRED)
  # ament_lint_auto_find_test_dependencies()
  # find_package(ament_cmake_gtest REQUIRED)

  # ament_add_gtest(test_history test/test_history.cpp)
  # target_link_libraries(test_history ${PROJECT_NAME})
endif()

ament_package()