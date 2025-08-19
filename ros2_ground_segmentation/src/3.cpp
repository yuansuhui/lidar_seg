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
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>       // PassThrough 滤波器
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <mutex> // Added for thread safety

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
  std::mutex odom_mutex_; // Added for thread safety


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
    Eigen::Matrix4f odom_T_lidar = Eigen::Matrix4f::Identity();
    tf2::Matrix3x3 R(q);
    Eigen::Matrix3f R_eigen;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        R_eigen(i, j) = R[i][j];
      }
    }

    // 赋值到 block
    odom_T_lidar.block<3,3>(0,0) = R_eigen;
    
    // odom_T_lidar.block<3,3>(0,0) << R, R[1], R[2],
    //                                 R[1], R[1][1], R[1][2],
    //                                 R[2], R[2][1], R[2][2];

    odom_T_lidar(0,3) = msg->pose.pose.position.x;
    odom_T_lidar(1,3) = msg->pose.pose.position.y;
    odom_T_lidar(2,3) = msg->pose.pose.position.z;

    // 存储到成员变量中，供点云回调时使用
    std::lock_guard<std::mutex> lock(odom_mutex_);
    latest_odom_pose_ = odom_T_lidar;
    // RCLCPP_INFO(this->get_logger(), "Odom time: %d.%09u",
    //           msg->header.stamp.sec,
    //           msg->header.stamp.nanosec);
  }

void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg)
{
  if (!latest_odom_pose_.isZero(1e-6)) {
  // RCLCPP_INFO(this->get_logger(), "point time: %d.%09u",
  //           cloud_msg->header.stamp.sec,
  //           cloud_msg->header.stamp.nanosec);

  bool map_cleared_ = false;

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*cloud_msg, *cloud);
  // std::cout << "cloud num: " << cloud->points.size() << std::endl;
  pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
  voxel_filter.setInputCloud(cloud);
  voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);  // 体素大小 (m)，根据需要调整

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
  voxel_filter.filter(*cloud_filtered);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZI>());
  for (const auto& pt : cloud_filtered->points) {
    if (!std::isfinite(pt.x) ||!std::isfinite(pt.y) ||!std::isfinite(pt.z)) continue;
    // 优化：缩小手动滤波的范围以减少计算量
    if (pt.z  > -2.0 && pt.z < 0.2 && pt.x<10 && pt.x>-10 && pt.y<10 && pt.y>-10) {
      cloud_copy->points.push_back(pt);
    }
  } 
  cloud_copy->width = cloud_copy->points.size();
  cloud_copy->height = 1;
  cloud_copy->is_dense = true;
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());

  // 将 latest_odom_pose_ 从 double 类型转换为 float 类型，并使用互斥锁确保线程安全
  Eigen::Matrix4f transform;
  {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      transform = latest_odom_pose_.cast<float>();

  }

  // 变换点云
  pcl::transformPointCloud(*cloud_copy, *transformed_cloud, transform);
  auto start = std::chrono::high_resolution_clock::now();  // 开始时间
  // std::cout << "cloud_copy num: " << cloud_copy->points.size() << std::endl;
  // ------------------------------
  // Step 1: LMedS 拟合平面初步提取地面 inliers
  // ------------------------------
  pcl::PassThrough<PointInT> pass;
  pass.setInputCloud(transformed_cloud);
  pass.setFilterFieldName("z");
  // 优化：缩小 PassThrough 滤波的范围
  pass.setFilterLimits(odom_z-2.0, odom_z+0.2); 
  pass.filter(*transformed_cloud);
  pcl::SACSegmentation<PointInT> seg;
  pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_LMEDS); 
  // 优化：减少迭代次数和距离阈值
  seg.setDistanceThreshold(0.07);
  seg.setMaxIterations(20);
  seg.setInputCloud(transformed_cloud);
  seg.segment(*ground_inliers, *coefficients);

  if (ground_inliers->indices.empty()) {
      std::cout << "latest_odom_pose_:\n" 
                << latest_odom_pose_.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", "\n", "[", "]"))
                << std::endl;
    RCLCPP_WARN(this->get_logger(), "No ground plane found by LMedS.");
    auto cloud_non_ground_rgb = convertToColoredCloud(cloud, 255, 0, 0);
    publishCloud(cloud_non_ground_rgb, cloud_msg->header, pub_non_ground_);
    return;
  }
  auto end = std::chrono::high_resolution_clock::now();    // 结束时间

  // 计算耗时（毫秒）
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  // std::cout << "seg time: " << duration.count() << " ms" << std::endl;

  pcl::PointCloud<PointInT>::Ptr cloud_ground(new pcl::PointCloud<PointInT>);
  pcl::ExtractIndices<PointInT> extract;
  extract.setInputCloud(transformed_cloud);

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

  Eigen::Vector3f x = A.householderQr().solve(b); // 解线性最小二乘问题

  float a = x(0), b_ = x(1), d = x(2);
  float c = 1.0f; // z 系数设为 1，得到 plane: ax + by + z + d = 0
  Eigen::Vector3f normal(a, b_, 1.0f);
  normal.normalize();
  Eigen::Vector3f odo_z(0.0, 0.0, odom_z);

  float dot = normal.dot(odo_z);  // 点积
  float cos_theta = dot / (normal.norm() * odo_z.norm());
  
  // 防止数值误差导致 cos_theta 超过 [-1,1]
  cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));

  float angle = std::acos(cos_theta);  // 弧度
  float angle_deg = angle * 180.0f / M_PI; // 转换为角度
  if(angle_deg>165&&angle_deg<180){
    angle_deg = 180-angle_deg;
  }
  // std::cout << "夹角（角度）: " << odom_z << std::endl;
  std::cout << "2: " << abs(180-angle_deg) << std::endl;
  // RCLCPP_INFO(this->get_logger(), "Refined Plane: a=%.3f, b=%.3f, c=%.3f, d=%.3f", a, b_, c, d);

  // ------------------------------
  // Step 3: 基于 refined plane 分割点云（更精准的地面/非地面）
  // ------------------------------
  pcl::PointCloud<PointInT>::Ptr cloud_refined_ground(new pcl::PointCloud<PointInT>);
  pcl::PointCloud<PointInT>::Ptr cloud_refined_non_ground(new pcl::PointCloud<PointInT>);
  pcl::PointCloud<PointInT>::Ptr cloud_refined_non_ground2(new pcl::PointCloud<PointInT>);
  cloud_refined_non_ground2->points.clear();
  cloud_refined_non_ground->points.clear();
  cloud_refined_ground->points.clear();
  for (const auto& pt : transformed_cloud->points) {
    float dist = std::fabs(a * pt.x + b_ * pt.y + pt.z + d) / std::sqrt(a * a + b_ * b_ + c * c);
    
    if (dist < 0.1f && angle_deg<30) { // 精修平面距离阈值
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
  cloud_refined_non_ground2->points.clear();
  cloud_refined_non_ground->points.clear();
  cloud_refined_ground->points.clear();
  // auto end = std::chrono::high_resolution_clock::now();    // 结束时间

  // // 计算耗时（毫秒）
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  // std::cout << "seg time: " << duration.count() << " ms" << std::endl;
  }else{
    return;
  }
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
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), 
        std::thread::hardware_concurrency()
    );

    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
  // rclcpp::spin(node);
  // rclcpp::shutdown();
  // return 0;
}
