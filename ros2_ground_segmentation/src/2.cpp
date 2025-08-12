// ground_segmentation_node.cpp
#include <memory>
#define DEG2RAD(x) ((x) * M_PI / 180.0)

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/features/normal_3d.h>
#include <pcl/common/angles.h>  // pcl::rad2deg
#include <pcl/surface/mls.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
// 加载带颜色点类型
#include <pcl/point_types.h>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl/common/transforms.h>
#include <cmath>

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
      "/rslidar_points", 50,
      std::bind(&GroundSegmentationNode::pointcloud_callback, this, std::placeholders::_1));
    // 订阅里程计话题
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/Odometry", 50,
      std::bind(&GroundSegmentationNode::odom_callback, this, std::placeholders::_1));
    // 发布带颜色地面点云
    pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("ground_points_colored", 10);
    // 发布带颜色非地面点云
    pub_non_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("non_ground_points_colored", 10);
    // timer_ = this->create_wall_timer(
    //   std::chrono::milliseconds(500),  // 每 0.5 秒检查一次
    //   std::bind(&GroundSegmentationNode::timer_callback, this));
    //     RCLCPP_INFO(this->get_logger(), "Ground Segmentation Node started.");
  }

private:
  double odom_z;
  double odom_x;
  double odom_y;
  // rclcpp::Time last_data_time_  = this->get_clock()->now(); 
  rclcpp::TimerBase::SharedPtr timer_;
  bool map_cleared_ = false;
  geometry_msgs::msg::Point current_position_;
  geometry_msgs::msg::Quaternion current_orientation_;
  Eigen::Matrix4f latest_odom_pose_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  // void timer_callback()
  // {
  //   auto now = this->get_clock()->now();
  //   double elapsed = (now - last_data_time_).seconds();

  //   if (elapsed > 5.0 && !map_cleared_) {
  //     RCLCPP_WARN(this->get_logger(), "No pointcloud received in 5s, clearing clouds.");

  //     // cloud_refined_ground->clear();
  //     // cloud_refined_non_ground->clear();

  //     // 发布空点云
  //     auto empty_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  //     std_msgs::msg::Header header;
  //     header.stamp = this->now();
  //     header.frame_id = "camera_init"; 
  //     // auto header = header;
      

  //     publishCloud(empty_cloud, header, pub_ground_);
  //     publishCloud(empty_cloud, header, pub_non_ground_);

  //     map_cleared_ = true;
  //   }
  // }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
  //   if (!msg ) {
  //   RCLCPP_INFO(this->get_logger(), "Received empty cloud_msg!");
  //   return;
  // }
      current_position_ = msg->pose.pose.position;
      current_orientation_ = msg->pose.pose.orientation;
      odom_z = current_position_.z;
      odom_x = current_position_.x;
      odom_y = current_position_.y;

    tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w);
    
    tf2::Matrix3x3 R(q);

    Eigen::Matrix4f odom_T_lidar = Eigen::Matrix4f::Identity();
    odom_T_lidar.block<3,3>(0,0) << R[0][0], R[0][1], R[0][2],
                                    R[1][0], R[1][1], R[1][2],
                                    R[2][0], R[2][1], R[2][2];

    odom_T_lidar(0,3) = msg->pose.pose.position.x;
    odom_T_lidar(1,3) = msg->pose.pose.position.y;
    odom_T_lidar(2,3) = msg->pose.pose.position.z;

    // 存储到成员变量中，供点云回调时使用
    latest_odom_pose_ = odom_T_lidar;
  }

void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
{
  bool map_cleared_ = false;

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*cloud_msg, *cloud);
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud2(new pcl::PointCloud<pcl::PointXYZI>());
  for (const auto& pt2 : cloud->points) {
  if (!std::isfinite(pt2.x) || !std::isfinite(pt2.y) || !std::isfinite(pt2.z)) continue;
  //   // 移除距离小于 0.8 的点
    if (((pt2.x ) <1.2 && (pt2.x ) > -1.2)) continue;

    transformed_cloud2->points.push_back(pt2);
  } 
  transformed_cloud2->width = transformed_cloud2->points.size();
  transformed_cloud2->height = 1;
  transformed_cloud2->is_dense = true;

  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());

  // 将 latest_odom_pose_ 从 double 类型转换为 float 类型
  Eigen::Matrix4f transform = latest_odom_pose_.cast<float>();

  // 变换点云
  pcl::transformPointCloud(*transformed_cloud2, *transformed_cloud, transform);
 // 将 msg 转换为 PCL 点云

  // 创建一个新的点云，用于保存筛选后的点
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
  for (const auto& pt : transformed_cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    cloud_filtered->points.push_back(pt);
  } 

  if (cloud_filtered->empty()) {
    RCLCPP_WARN(this->get_logger(), "Received empty point cloud");
    return;
  }

  pcl::PointCloud<PointInT>::Ptr cloud_filtered_z(new pcl::PointCloud<PointInT>);
  for (const auto& pt : cloud_filtered->points) {
    if (pt.z - odom_z > -2 && pt.z - odom_z < -0.1) {
      cloud_filtered_z->points.push_back(pt);
    }
  }
  // RCLCPP_INFO(this->get_logger(), "处理完成 %zu 个点", cloud_filtered_z->points.size());
  if (cloud_filtered_z->empty()) {
    // RCLCPP_WARN(this->get_logger(), "Filtered point cloud has no points in Z range");
    return;
  }

  cloud_filtered_z->width = cloud_filtered_z->points.size();
  cloud_filtered_z->height = 1;
  cloud_filtered_z->is_dense = true;

  // ------------------------------
  // Step 1: LMedS 拟合平面初步提取地面 inliers
  // ------------------------------
  pcl::SACSegmentation<PointInT> seg;
  pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_LMEDS); // 
  seg.setDistanceThreshold(0.07);
  seg.setMaxIterations(500);
  seg.setInputCloud(cloud_filtered_z);
  seg.segment(*ground_inliers, *coefficients);

  if (ground_inliers->indices.empty()) {
    RCLCPP_WARN(this->get_logger(), "No ground plane found by LMedS.");
    auto cloud_non_ground_rgb = convertToColoredCloud(cloud, 255, 0, 0);
    publishCloud(cloud_non_ground_rgb, cloud_msg->header, pub_non_ground_);
    return;
  }


  pcl::PointCloud<PointInT>::Ptr cloud_ground(new pcl::PointCloud<PointInT>);
  pcl::ExtractIndices<PointInT> extract;
  extract.setInputCloud(cloud_filtered_z);
  extract.setIndices(ground_inliers);
  extract.setNegative(false);
  extract.filter(*cloud_ground);

  Eigen::MatrixXf A(cloud_ground->points.size(), 3);
  Eigen::VectorXf b(cloud_ground->points.size());
  for (size_t i = 0; i < cloud_ground->points.size(); ++i) {
    A(i, 0) = cloud_ground->points[i].x;
    A(i, 1) = cloud_ground->points[i].y;
    A(i, 2) = 1.0f;
    b(i) = -cloud_ground->points[i].z;
  }

  Eigen::Vector3f x = A.colPivHouseholderQr().solve(b); // 解线性最小二乘问题

  float a = x(0), b_ = x(1), d = x(2);
  float c = 1.0f; // z 系数设为 1，得到 plane: ax + by + z + d = 0

  // RCLCPP_INFO(this->get_logger(), "Refined Plane: a=%.3f, b=%.3f, c=%.3f, d=%.3f", a, b_, c, d);

  // ------------------------------
  // Step 3: 基于 refined plane 分割点云（更精准的地面/非地面）
  // ------------------------------
  pcl::PointCloud<PointInT>::Ptr cloud_refined_ground(new pcl::PointCloud<PointInT>);
  pcl::PointCloud<PointInT>::Ptr cloud_refined_non_ground(new pcl::PointCloud<PointInT>);

  for (const auto& pt : transformed_cloud->points) {
    float dist = std::fabs(a * pt.x + b_ * pt.y + pt.z + d) / std::sqrt(a * a + b_ * b_ + c * c);
    // float r = 0.0;
    // if(pt.y-odom_y>-2&&pt.y-odom_y<2&&pt.x-odom_x>9){
    //   r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
    // }
    
    // float adaptive_thresh;
    // if(r>10){
    //   adaptive_thresh = 0.1 + 0.03 * r;
    // }else{
    //   adaptive_thresh = 0.1 ;
    // }
    
    if (dist < 0.1f) { // 精修平面距离阈值
      cloud_refined_ground->points.push_back(pt);
    } else {
      cloud_refined_non_ground->points.push_back(pt);
    }
  }
  for(const auto& pt : cloud_refined_non_ground->points){
    if(pt.x-odom_x>-1.2&&pt.x-odom_x<1.2){
      continue;
    }else{
      cloud_refined_non_ground2->points.push_back(pt);
    }
  }

  auto cloud_ground_rgb = convertToColoredCloud(cloud_refined_ground, 0, 255, 0);
  auto cloud_non_ground_rgb = convertToColoredCloud(cloud_refined_non_ground2, 255, 0, 0);

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
    output.header.frame_id = "camera_init";
    pub->publish(output);
    // RCLCPP_INFO(this->get_logger(), "Published point cloud with %lu points.", cloud->points.size());
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
