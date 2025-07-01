// ground_segmentation_node.cpp
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
// 加载带颜色点类型
#include <pcl/point_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

using PointInT = pcl::PointXYZI;    // 原始点云类型
using PointRGBT = pcl::PointXYZRGB; // 带颜色点云类型

class GroundSegmentationNode : public rclcpp::Node
{
public:
  GroundSegmentationNode() : Node("ground_segmentation_node")
  {
    // 订阅点云
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rslidar_points", 10,
      std::bind(&GroundSegmentationNode::pointcloud_callback, this, std::placeholders::_1));

    // 发布带颜色地面点云
    pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("ground_points_colored", 10);
    // 发布带颜色非地面点云
    pub_non_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("non_ground_points_colored", 10);

    RCLCPP_INFO(this->get_logger(), "Ground Segmentation Node started.");
  }

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
  {
    // 转成 PCL 原始点云格式
    pcl::PointCloud<PointInT>::Ptr cloud(new pcl::PointCloud<PointInT>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (cloud->empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty point cloud");
      return;
    }

    // 创建 SACSegmentation 对象进行平面拟合
    pcl::SACSegmentation<PointInT> seg;
    pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.1); // 距离阈值，可调节，单位米
    seg.setInputCloud(cloud);
    seg.segment(*ground_inliers, *coefficients);

    if (ground_inliers->indices.empty()) {
      RCLCPP_WARN(this->get_logger(), "No ground plane found.");
      // 全部视为非地面，赋红色发布
      auto cloud_non_ground_rgb = convertToColoredCloud(cloud, 255, 0, 0);
      publishCloud(cloud_non_ground_rgb, cloud_msg->header, pub_non_ground_);
      return;
    }

    // 根据inliers提取地面点云索引，用于赋绿
    pcl::ExtractIndices<PointInT> extract;
    pcl::PointCloud<PointInT>::Ptr cloud_ground(new pcl::PointCloud<PointInT>);
    extract.setInputCloud(cloud);
    extract.setIndices(ground_inliers);
    extract.setNegative(false);
    extract.filter(*cloud_ground);

    // 非地面点云
    pcl::PointCloud<PointInT>::Ptr cloud_non_ground(new pcl::PointCloud<PointInT>);
    extract.setNegative(true);
    extract.filter(*cloud_non_ground);

    // 转成带颜色点云，赋颜色
    auto cloud_ground_rgb = convertToColoredCloud(cloud_ground, 0, 255, 0);     // 绿地面
    auto cloud_non_ground_rgb = convertToColoredCloud(cloud_non_ground, 255, 0, 0); // 红非地面

    // 发布
    publishCloud(cloud_ground_rgb, cloud_msg->header, pub_ground_);
    publishCloud(cloud_non_ground_rgb, cloud_msg->header, pub_non_ground_);
  }

  // 将不带颜色的点云转成带颜色点云并赋值RGB
  pcl::PointCloud<PointRGBT>::Ptr convertToColoredCloud(
    const pcl::PointCloud<PointInT>::Ptr & input_cloud,
    uint8_t r, uint8_t g, uint8_t b)
  {
    pcl::PointCloud<PointRGBT>::Ptr colored_cloud(new pcl::PointCloud<PointRGBT>);
    colored_cloud->points.reserve(input_cloud->points.size());

    for (const auto &pt : input_cloud->points) {
      PointRGBT p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = pt.z;
      p.r = r;
      p.g = g;
      p.b = b;
      colored_cloud->points.push_back(p);
    }

    colored_cloud->width = static_cast<uint32_t>(colored_cloud->points.size());
    colored_cloud->height = 1;
    colored_cloud->is_dense = input_cloud->is_dense;

    return colored_cloud;
  }

  void publishCloud(
    pcl::PointCloud<PointRGBT>::Ptr cloud,
    const std_msgs::msg::Header & header,
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub)
  {
    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header = header;
    pub->publish(output);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_non_ground_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GroundSegmentationNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
