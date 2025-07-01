cmake_minimum_required(VERSION 3.5)
project(ground_segmentation)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation sample_consensus search)

include_directories(include ${PCL_INCLUDE_DIRS})

add_executable(ground_segmentation_node src/ground_segmentation_node.cpp)
ament_target_dependencies(ground_segmentation_node
  rclcpp sensor_msgs pcl_conversions
)

target_link_libraries(ground_segmentation_node ${PCL_LIBRARIES})

install(TARGETS
  ground_segmentation_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
