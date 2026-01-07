#ifndef NAV2_RL_CONTROLLER__RL_CONTROLLER_HPP_
#define NAV2_RL_CONTROLLER__RL_CONTROLLER_HPP_

#include <deque>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <tuple>

#include "nav2_core/controller.hpp"
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp/rclcpp.hpp>
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/path.hpp"

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace nav2_rl_controller
{

class RLController : public nav2_core::Controller
{
public:
  RLController() = default;
  ~RLController() override = default;

  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  void cleanup() override;
  void activate() override;
  void deactivate() override;
  void reset();

  void setSpeedLimit(const double & speed_limit, const bool & percentage) override;

  void setPlan(const nav_msgs::msg::Path & path) override;

  geometry_msgs::msg::TwistStamped computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped & pose,
    const geometry_msgs::msg::Twist & velocity,
    nav2_core::GoalChecker * goal_checker) override;

protected:
  // 基于 costmap 计算 obs_min（20 维扇区距离），在 assembleObservation 中调用
  std::vector<float> computeObsFromCostmap(const geometry_msgs::msg::PoseStamped * pose);

  // 从已保存的全局路径中基于当前位姿选取下一个目标点并返回 target_cos, target_sin, target_distance
  std::tuple<double, double, double> computeTargetFromPlan(const geometry_msgs::msg::PoseStamped & current_pose);

  // 模型推理，返回二维的线速度和角速度
  std::vector<float> runModel(const std::vector<float> & input);

  // 组装一次完整的模型输入
  std::vector<float> assembleObservation(
    const geometry_msgs::msg::PoseStamped * pose,
    const geometry_msgs::msg::Twist * vel,
    std::vector<float> & current_frame_out);
  
  // 成员变量
  rclcpp_lifecycle::LifecycleNode::WeakPtr node_;
  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::string plugin_name_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_ = nullptr;


  std::mutex history_mutex_;
  // 历史帧缓冲：保存已形成的完整帧（每帧 25 维：20 obs_min + 3 target + 2 last_action）
  // 内容由 computeVelocityCommands 在推理后加入（推理完成后会把模型输出回写到帧的最后两维）
  std::deque<std::vector<float>> history_frames_;

  // 历史帧数量（不包含当前帧），默认 50；实际传入模型的帧数为 history_length_ + 1（包含当前帧）
  size_t history_length_ = 50;

  // 每帧完整观测维度为 min_obs_dim_ + 3 + 2（通常为 25）
  size_t obs_dim_ = 25;

  // obs_min 的维度（默认 20）
  size_t min_obs_dim_ = 20;

  // 保存模型上一次输出（linear, angular）用于构造当前帧的最后两维
  geometry_msgs::msg::Twist last_action_;

  double last_obs_min_dist_ = 1e6;

  // 保存最新的全局路径（由 setPlan 设置），用于在 computeVelocityCommands 中选取目标点
  nav_msgs::msg::Path latest_plan_;
  std::mutex plan_mutex_;
  bool have_plan_ = false;

  // 扇区内射线数（默认每扇区采样 8 条射线），用于更稳健地估计最近障碍距离
  size_t rays_per_sector_ = 8;


  // Delay constructing Ort::Env until runModel() to avoid ABI compatibility issues during plugin configuration
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
  Ort::SessionOptions session_options_;
  // Lazy init controls for ONNX session
  std::mutex ort_mutex_;
  bool ort_failed_ = false; // if true, further attempts to init will be skipped
  // Model input size expected (1275)
  size_t model_input_size_ = 1275;


  // 外部可设置参数
  std::string model_path_ = "";
  double max_linear_speed_ = 0.5;
  double base_max_linear_speed_ = 0.5;
  double max_angular_speed_ = 1.0;
  double min_obs_distance_ = 0.2;
  // 路径稀疏化距离（米），每sparse_path_distance米保留一个路径点
  double sparse_path_distance_ = 2.5;
  // debug模式
  bool debug = true;
  // 这几个并不能设置，ros2似乎不能设置 string 类型的参数
  std::string output_observations_file = "/home/unitree/nav2_gps/nav2_rl_controller/logs/observations.txt";
  std::string output_img_file = "/home/unitree/nav2_gps/nav2_rl_controller/logs/img.jpg";
  std::string output_compute_file = "/home/unitree/nav2_gps/nav2_rl_controller/logs/compute.txt";
  std::string output_model_run_file = "/home/unitree/nav2_gps/nav2_rl_controller/logs/model_run.txt";

  // 辅助函数：从四元数计算 yaw
  double yawFromQuat(const geometry_msgs::msg::Quaternion & q);

  // 辅助函数：路径稀疏化
  nav_msgs::msg::Path sparsePath(const nav_msgs::msg::Path & path, double distance_threshold);

  // 辅助函数：调试时将观测保存到文件
  void saveObservationToFile(const std::vector<float>& obs);
  
  // 辅助函数：调试时将障碍物距离绘制成图像并保存
  bool saveCostmapImage(const std::vector<float>& obs, int image_size);
};

}  // namespace nav2_rl_controller

#endif  // NAV2_RL_CONTROLLER__RL_CONTROLLER_HPP_