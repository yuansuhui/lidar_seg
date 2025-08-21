// ground_segmentation_node.cpp
#include <memory>
#define DEG2RAD(x) ((x) * M_PI / 180.0)

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/features/normal_3d.h>
#include <pcl/common/angles.h> 
#include <pcl/surface/mls.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl/common/transforms.h>
#include <cmath>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>       
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <mutex>
using PointInT = pcl::PointXYZI;    
using PointRGBT = pcl::PointXYZRGB; 
struct PointVote {
    pcl::PointXYZI point;
    int obstacle_count = 0;
};
class GroundSegmentationNode : public rclcpp::Node
{
public:
  GroundSegmentationNode() : Node("ground_segmentation_node")
  {
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rslidar_points", 50,
      std::bind(&GroundSegmentationNode::pointcloud_callback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/Odometry", 50,
      std::bind(&GroundSegmentationNode::odom_callback, this, std::placeholders::_1));

    pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("ground_points_colored", 10);
    pub_non_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("non_ground_points_colored", 10);
    pub_non_ground2_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("non_ground_points_colored2", 10);
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
  std::mutex odom_mutex_; 


  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
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
    double roll, pitch, yaw;
    R.getRPY(roll, pitch, yaw);  // 这里单位是弧度

    // RCLCPP_INFO(this->get_logger(), 
        // "Roll: %.3f, Pitch: %.3f, Yaw: %.3f (rad)", roll, pitch, yaw);

    double roll_deg = roll * 180.0 / M_PI;
    double pitch_deg = pitch * 180.0 / M_PI;
    double yaw_deg = yaw * 180.0 / M_PI;

    // RCLCPP_INFO(this->get_logger(), 
    //     "Roll: %.2f°, Pitch: %.2f°, Yaw: %.2f°", roll_deg, pitch_deg, yaw_deg);
    Eigen::Matrix3f R_eigen;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        R_eigen(i, j) = R[i][j];
      }
    }

    odom_T_lidar.block<3,3>(0,0) = R_eigen;
    
    // odom_T_lidar.block<3,3>(0,0) << R, R[1], R[2],
    //                                 R[1], R[1][1], R[1][2],
    //                                 R[2], R[2][1], R[2][2];

    odom_T_lidar(0,3) = msg->pose.pose.position.x;
    odom_T_lidar(1,3) = msg->pose.pose.position.y;
    odom_T_lidar(2,3) = msg->pose.pose.position.z;

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
  voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f); 

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
  voxel_filter.filter(*cloud_filtered);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_copy(new pcl::PointCloud<pcl::PointXYZI>());
  for (const auto& pt : cloud_filtered->points) {
    if (!std::isfinite(pt.x) ||!std::isfinite(pt.y) ||!std::isfinite(pt.z)) continue;
    if (pt.z  > -2.0 && pt.z < 0.2 && pt.x<10 && pt.x>-10 && pt.y<10 && pt.y>-10) {
      cloud_copy->points.push_back(pt);
    }
  } 
  cloud_copy->width = cloud_copy->points.size();
  cloud_copy->height = 1;
  cloud_copy->is_dense = true;
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());

  Eigen::Matrix4f transform;
  {
      std::lock_guard<std::mutex> lock(odom_mutex_);
      transform = latest_odom_pose_.cast<float>();

  }

  pcl::transformPointCloud(*cloud_copy, *transformed_cloud, transform);
  auto start = std::chrono::high_resolution_clock::now(); 
  // std::cout << "cloud_copy num: " << cloud_copy->points.size() << std::endl;
  // ------------------------------
  // ------------------------------
  // pcl::PassThrough<PointInT> pass;
  // pass.setInputCloud(transformed_cloud);
  // pass.setFilterFieldName("z");
  // pass.setFilterLimits(odom_z-2.0, odom_z+0.2); 
  // pass.filter(*transformed_cloud);
  pcl::SACSegmentation<PointInT> seg;
  pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_LMEDS); 
  seg.setDistanceThreshold(0.2);
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
  auto end = std::chrono::high_resolution_clock::now();  

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  // std::cout << "seg time: " << duration.count() << " ms" << std::endl;

  pcl::PointCloud<PointInT>::Ptr cloud_ground(new pcl::PointCloud<PointInT>);
  pcl::PointCloud<PointInT>::Ptr non_cloud_ground(new pcl::PointCloud<PointInT>);
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

  Eigen::Vector3f x = A.householderQr().solve(b);

  float a = x(0), b_ = x(1), d = x(2);
  float c = 1.0f; //plane: ax + by + z + d = 0
  Eigen::Vector3f normal(a, b_, 1.0f);
  normal.normalize();
  Eigen::Vector3f odo_z(0.0, 0.0, odom_z);

  float dot = normal.dot(odo_z);  
  float cos_theta = dot / (normal.norm() * odo_z.norm());
  
  cos_theta = std::max(-1.0f, std::min(1.0f, cos_theta));

  float angle = std::acos(cos_theta);  
  float angle_deg = angle * 180.0f / M_PI; 
  if(angle_deg>165&&angle_deg<180){
    angle_deg = 180-angle_deg;
  }
  
  // RCLCPP_INFO(this->get_logger(), "Refined Plane: a=%.3f, b=%.3f, c=%.3f, d=%.3f", a, b_, c, d);

  // ------------------------------
  pcl::PointCloud<PointInT>::Ptr cloud_refined_ground(new pcl::PointCloud<PointInT>);
  pcl::PointCloud<PointInT>::Ptr cloud_refined_non_ground(new pcl::PointCloud<PointInT>);
  pcl::PointCloud<PointInT>::Ptr cloud_refined_non_ground2(new pcl::PointCloud<PointInT>);
  cloud_refined_non_ground2->points.clear();
  cloud_refined_non_ground->points.clear();
  cloud_refined_ground->points.clear();

  for (const auto& pt : transformed_cloud->points) {
    float dist = std::fabs(a * pt.x + b_ * pt.y + pt.z + d) / std::sqrt(a * a + b_ * b_ + c * c);
    // std::cout << "abs(180-angle_deg): " << abs(180-angle_deg) << std::endl;
    // std::cout << "dist: " << dist<< std::endl;
    if (dist < 0.1f && angle_deg<30) { 
      cloud_refined_ground->points.push_back(pt);
    } else {
      cloud_refined_non_ground->points.push_back(pt);
    }
  }
  pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
  pcl::PointCloud<PointInT>::Ptr cloud_refined_non_ground2_new(new pcl::PointCloud<PointInT>);
  pcl::PointCloud<PointInT>::Ptr cloud_refined_non_ground2_new2(new pcl::PointCloud<PointInT>);
  kdtree.setInputCloud(cloud_refined_non_ground);
  // std::cout << "cloud_refined_ground: " << cloud_refined_ground->points.size()<< std::endl;
  for (size_t i = 0; i < cloud_refined_non_ground->points.size(); ++i) {
      auto pt = cloud_refined_non_ground->points[i];


      std::vector<int> indices;
      std::vector<float> sqr_distances;
      float radius = 0.2f;

      if (kdtree.radiusSearch(pt, radius, indices, sqr_distances) > 0) {
          float z_min = 1e9, z_max = -1e9;
          for (int idx : indices) {
              float z = cloud_refined_non_ground->points[idx].z;
                if (z <= z_min) z_min = z;
                if (z > z_max) z_max = z;

          }

          float dz = z_max - z_min;
          if (dz < 0.06f) {
              cloud_refined_ground->points.push_back(pt);
          } else {
              cloud_refined_non_ground2_new->points.push_back(pt);
          }
      }
  }
  // cloud_refined_non_ground.swap(cloud_refined_non_ground2_new);
// std::cout << "cloud_refined_ground22222: " << cloud_refined_ground->points.size()<< std::endl;
  for(const auto& pt : cloud_refined_non_ground2_new->points){
    if(pt.x-odom_x>-1.0&&pt.x-odom_x<1.0){
      cloud_refined_ground->points.push_back(pt);
    }else{
      cloud_refined_non_ground2_new2->points.push_back(pt);
    }
  }
  cloud_refined_non_ground.swap(cloud_refined_non_ground2_new2);
  auto cloud_ground_rgb = convertToColoredCloud(cloud_refined_ground, 0, 255, 0);
  auto cloud_non_ground_rgb2 = convertToColoredCloud(transformed_cloud, 100, 100, 200);
  auto cloud_non_ground_rgb = convertToColoredCloud(cloud_refined_non_ground, 255, 0, 0);

  publishCloud(cloud_ground_rgb, cloud_msg->header, pub_ground_);
  publishCloud(cloud_non_ground_rgb, cloud_msg->header, pub_non_ground_);
  publishCloud(cloud_non_ground_rgb2, cloud_msg->header, pub_non_ground2_);
  cloud_refined_non_ground2->points.clear();
  cloud_refined_non_ground->points.clear();
  cloud_refined_ground->points.clear();
  // auto end = std::chrono::high_resolution_clock::now();   

  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  // std::cout << "seg time: " << duration.count() << " ms" << std::endl;
  }else{
    std::cout << "latest_odom_pose_:\n" 
          << latest_odom_pose_.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", "\n", "[", "]"))
          << std::endl;
    return;
  }
}



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
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_non_ground2_;
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

