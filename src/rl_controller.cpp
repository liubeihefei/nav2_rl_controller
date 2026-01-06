#include "nav2_rl_controller/rl_controller.hpp"
#include "pluginlib/class_list_macros.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <tuple>
#include <deque>
#include <string>
#include <fstream>

using namespace std::chrono_literals;

/*
 * RL Controller 插件说明（中文）：
 * - 输入历史：使用最近 history_length_ 帧已完成的完整帧（每帧 25 维）加上当前帧，共 history_length_ + 1 帧
 *   每帧格式：[ obs_min(20) | target_cos (1) | target_sin (1) | target_dist (1) | last_action_linear (1) | last_action_angular (1) ]
 * - 数据流：
 *   1) 在需要时，控制器使用内部的 costmap 来计算 20 维扇区化 obs_min（不再订阅外部话题）
 *   2) computeVelocityCommands 被调用时：
 *        - 用 costmap 衍生的 obs_min + 基于当前 pose 与已保存全局路径计算的 target(3) + last_action（上一次模型输出）
 *          构造当前帧（25 维），将其与历史完整帧拼接成模型输入（扁平化）
 *        - 将输入发送给 ONNX 模型进行推理，得到新的动作输出（linear, angular）
 *        - 更新 last_action，并把模型输出写回当前帧的最后两维，然后把当前帧加入历史帧缓冲
 *   3) 下次推理重复上述过程，历史帧的最后两维始终保存了当时模型的输出（即回执）
 * - 设计要点：历史帧存储完整帧（含模型当时输出），以保证历史信息反映推理时刻的动作反馈；
 *   target 只在推理时使用当前位姿计算，历史帧不会被重新计算 target。
 */

namespace nav2_rl_controller
{
// 注册，自己不触发，初始化各个参数
void RLController::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
    std::string name, std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent;
  plugin_name_ = name;
  tf_ = tf;
  costmap_ros_ = costmap_ros;
  if (costmap_ros_) {
      costmap_ = costmap_ros_->getCostmap();
  }

  // 加载参数
  auto node = node_.lock();
  if (node) RCLCPP_INFO(node->get_logger(), "RLController::configure start (%s)", plugin_name_.c_str());
  node->get_parameter_or("model_path", model_path_, std::string("/path/to/SAC_actor.onnx"));
  // history_length 参数以整数读取再赋值给 size_t 成员，避免 rclcpp 参数模板歧义
  int history_length_param = static_cast<int>(history_length_);
  node->get_parameter_or("history_length", history_length_param, history_length_param);
  history_length_ = static_cast<size_t>(history_length_param);

  // obs_min 的维度（默认 20）
  int min_obs_dim_param = static_cast<int>(min_obs_dim_);
  node->get_parameter_or("min_obs_dim", min_obs_dim_param, min_obs_dim_param);
  min_obs_dim_ = static_cast<size_t>(min_obs_dim_param);
  // 每帧完整维度 = min_obs_dim + 3(target) + 2(last_action)
  obs_dim_ = static_cast<size_t>(min_obs_dim_) + 3 + 2; // 25
  // 模型输入大小 = (history_length + 1) * obs_dim（历史 N 帧 + 当前 1 帧）
  model_input_size_ = (history_length_ + 1) * obs_dim_;

  node->get_parameter_or("max_linear_speed", max_linear_speed_, max_linear_speed_);
  base_max_linear_speed_ = max_linear_speed_;
  node->get_parameter_or("max_angular_speed", max_angular_speed_, max_angular_speed_);
  node->get_parameter_or("min_obs_distance", min_obs_distance_, min_obs_distance_);
  // 每扇区内部的射线数（用于更稳健的扇区最小距离估计，默认 8）
  int rays_per_sector_param = static_cast<int>(rays_per_sector_);
  node->get_parameter_or("rays_per_sector", rays_per_sector_param, rays_per_sector_param);
  rays_per_sector_ = static_cast<size_t>(rays_per_sector_param);
  // 路径稀疏化距离（米），默认2.5米
  node->get_parameter_or("sparse_path_distance", sparse_path_distance_, sparse_path_distance_);
  // debug模式
  node->get_parameter_or("debug", debug, false);

  // Configure ONNX session options; delay actual session creation until first inference
  session_options_.SetIntraOpNumThreads(1);
  // Use legacy macro ORT_ENABLE_ALL for compatibility across ONNX Runtime versions
  session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
  // Session will be created lazily in runModel() to avoid crash during startup
  if (node) RCLCPP_INFO(node->get_logger(), "RLController::configure done (%s)", plugin_name_.c_str());
  // Initialize Ort::Env here (after configuration logs) so early global construction does not happen
  try {
    std::lock_guard<std::mutex> lock(ort_mutex_);
    if (!ort_env_ && !ort_failed_) {
      ort_env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, plugin_name_.c_str()));
      if (node) RCLCPP_INFO(node->get_logger(), "Ort::Env initialized in configure for %s", plugin_name_.c_str());
    }
  } catch (const std::exception & e) {
    if (node) RCLCPP_ERROR(node->get_logger(), "Failed to initialize Ort::Env in configure: %s", e.what());
    ort_failed_ = true;
  } catch (...) {
    if (node) RCLCPP_ERROR(node->get_logger(), "Unknown failure initializing Ort::Env in configure");
    ort_failed_ = true;
  }
}

// 清理，自己不触发，清空所有历史观测和重置模型
void RLController::cleanup()
{
  history_frames_.clear();
  ort_session_.reset();
  // Also release Ort::Env to avoid lingering global state
  std::lock_guard<std::mutex> lock(ort_mutex_);
  ort_env_.reset();
}

// 激活，暂不实现
void RLController::activate()
{
  // No-op for now
}

// 失效，暂不实现
void RLController::deactivate()
{
  // No-op for now
}

// 重置，自己不触发，清空所有历史观测
void RLController::reset()
{
  std::lock_guard<std::mutex> lock(history_mutex_);
  history_frames_.clear();
}

// 获得全局规划的路径，自己不调用（官方插件也是如此）
void RLController::setPlan(const nav_msgs::msg::Path & path)
{
  // 获取全局规划的路径并进行稀疏化处理
  std::lock_guard<std::mutex> lock(plan_mutex_);
  if (path.poses.empty()) {
    latest_plan_ = path;
    have_plan_ = false;
    return;
  }
  
  // 对路径进行稀疏化处理
  latest_plan_ = sparsePath(path, sparse_path_distance_);
  have_plan_ = !latest_plan_.poses.empty();
  
  auto node = node_.lock();
  if (node) {
    RCLCPP_DEBUG(node->get_logger(), "Path sparsified: %zu -> %zu poses (distance threshold: %.2f m)", 
                  path.poses.size(), latest_plan_.poses.size(), sparse_path_distance_);
  }
}

// 设置速度上限，目前没用，后面配合参数进行线速度控制
void RLController::setSpeedLimit(const double & speed_limit, const bool & percentage)
{
  if (speed_limit == nav2_costmap_2d::NO_SPEED_LIMIT) {
    max_linear_speed_ = base_max_linear_speed_;
  } else {
    if (percentage) {
      max_linear_speed_ = base_max_linear_speed_ * speed_limit / 100.0;
    } else {
      max_linear_speed_ = speed_limit;
    }
  }
}

// 计算一次速度控制指令
geometry_msgs::msg::TwistStamped RLController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & velocity,
  nav2_core::GoalChecker * /*goal_checker*/)
{
  // 计算并返回控制命令：
  // 1) 组装最近历史的输入（50 × 25 + 25 = 1275）
  // 2) 调用 ONNX 模型推理
  // 3) 对输出做速度限幅和碰撞停止等安全检测
  auto node = node_.lock();
  geometry_msgs::msg::TwistStamped cmd_out;
  cmd_out.header.stamp = node ? node->now() : rclcpp::Time(0);

  try {
    // 构造当前输入帧并获取完整扁平化的模型输入
    std::vector<float> current_frame;
    std::vector<float> input = assembleObservation(&pose, &velocity, current_frame);
    // 注意：assembleObservation现在总是返回有效输入（历史不足时用零填充），不再返回空向量
    if (input.size() != model_input_size_) {
      if (node) RCLCPP_WARN(node->get_logger(), "Input size mismatch: %zu (expected %zu)", input.size(), model_input_size_);
      // return zero cmd on mismatch
      cmd_out.twist.linear.x = 0.0; cmd_out.twist.angular.z = 0.0;
      return cmd_out;
    }

    // Run model
    std::vector<float> output = runModel(input);
    // 打开文件（追加模式）
    std::ofstream outfile("/home/unitree/nav2_gps/nav2_rl_controller/output.txt", std::ios_base::app);
    if (output.size() == 2)
      outfile << output[0] << " " << output[1] << std::endl;
    else
      outfile << "模型输出大小不对：" << output.size() << std::endl;
    
    outfile.close();
    

    if (output.size() >= 2) {
        double lin = output[0];
        double ang = output[1];
        // enforce speed limits
        lin = std::clamp(lin, -max_linear_speed_, max_linear_speed_);
        ang = std::clamp(ang, -max_angular_speed_, max_angular_speed_);

        // Safety: if closest obstacle is too close, stop
        if (last_obs_min_dist_ < min_obs_distance_) {
            if (node) RCLCPP_WARN(node->get_logger(), "Obstacle too close (%.3f < %.3f), stopping" , last_obs_min_dist_, min_obs_distance_);
            lin = 0.0;
            ang = 0.0;
        }

        cmd_out.twist.linear.x = lin;
        cmd_out.twist.angular.z = ang;

        // Update last action
        last_action_.linear.x = lin;
        last_action_.angular.z = ang;

        // 将模型输出写回当前帧的最后两维，并将该帧保存到历史帧缓冲中（供后续推理使用）
        if (current_frame.size() == obs_dim_) {
            current_frame[min_obs_dim_ + 3] = static_cast<float>(lin);
            current_frame[min_obs_dim_ + 4] = static_cast<float>(ang);
            std::lock_guard<std::mutex> lock(history_mutex_);
            history_frames_.push_back(current_frame);
            while (history_frames_.size() > history_length_) history_frames_.pop_front();
        }
    } else {
        cmd_out.twist.linear.x = 0.0; cmd_out.twist.angular.z = 0.0;
    }

    return cmd_out;
  } catch (const std::exception & e) {
    if (node) RCLCPP_ERROR(node->get_logger(), "computeVelocityCommands exception: %s", e.what());
    cmd_out.twist.linear.x = 0.0; cmd_out.twist.angular.z = 0.0;
    return cmd_out;
  } catch (...) {
    if (node) RCLCPP_ERROR(node->get_logger(), "computeVelocityCommands unknown exception");
    cmd_out.twist.linear.x = 0.0; cmd_out.twist.angular.z = 0.0;
    return cmd_out;
  }
}

// 组装历史观测为模型输入
std::vector<float> RLController::assembleObservation(const geometry_msgs::msg::PoseStamped * pose, const geometry_msgs::msg::Twist * /*vel*/, std::vector<float> & current_frame_out)
{
  // 从 history_frames_中按照时间从旧到新取出最多 history_length_ 帧（每帧 25 维），不足则用当前帧（s0）填充
  // 组装输入：按照时间从旧到新拼接 history_length_ 帧历史（已包含模型输出）并在最后追加当前帧（尚未包含本次模型输出）
  // 当前帧由最新的 obs20（或回退）、基于 pose 与 path 计算的 target（3）以及 last_action_（2）组成
  // 历史不足时用当前帧填充（与episode开始时用s0填满历史队列的行为一致）
  std::vector<float> input(model_input_size_, 0.0f);
  std::lock_guard<std::mutex> lock(history_mutex_);

  size_t filled = history_frames_.size();
  size_t total_frames = history_length_ + 1;

  // 计算需要从历史中取多少帧，以及前端需补多少个 fallback 帧
  size_t take_from_history = std::min(filled, history_length_);
  size_t num_missing = (history_length_ > take_from_history) ? (history_length_ - take_from_history) : 0;

  size_t idx = 0;

  // 构造当前帧：obs20 + target(3) + last_action(2)
  std::vector<float> current_frame(obs_dim_, 0.0f);
  // obs20: 基于 costmap 扇区化计算（始终本地计算，不依赖订阅）
  std::vector<float> obs20 = computeObsFromCostmap(pose);
  for (size_t j = 0; j < min_obs_dim_ && j < obs20.size(); ++j) {
    current_frame[j] = obs20[j];
  }

  // target（三元组）基于当前 pose 与全局路径计算
  double tcos = 0.0, tsin = 0.0, tdist = 0.0;
  if (pose) std::tie(tcos, tsin, tdist) = computeTargetFromPlan(*pose);
  current_frame[min_obs_dim_ + 0] = static_cast<float>(tcos);
  current_frame[min_obs_dim_ + 1] = static_cast<float>(tsin);
  current_frame[min_obs_dim_ + 2] = static_cast<float>(tdist);

  // last_action（使用上一轮模型输出）
  current_frame[min_obs_dim_ + 3] = static_cast<float>(last_action_.linear.x);
  current_frame[min_obs_dim_ + 4] = static_cast<float>(last_action_.angular.z);

  // debug：保存最新的一帧输入到文件
  if(debug){
    saveObservationToFile(current_frame);
  }

  // 如果历史帧不足 history_length_：用当前帧（s0）填充前面的帧（与episode开始时保持一致）
  if (filled < history_length_) {
    // 计算需要填充的帧数
    size_t num_missing_frames = history_length_ - filled;
    // 用当前帧填充前面的帧（模拟episode开始时用s0填满历史队列）
    for (size_t m = 0; m < num_missing_frames; ++m) {
      for (size_t j = 0; j < obs_dim_ && idx < input.size(); ++j) {
          input[idx++] = current_frame[j];
      }
    }
    // 添加实际的历史帧（若某帧 malformed 则用 current_frame 填充并记录警告）
    for (size_t i = 0; i < filled; ++i) {
      const auto &frame = history_frames_[i];
      if (frame.size() != obs_dim_) {
        auto node = node_.lock();
        if (node) RCLCPP_WARN(node->get_logger(), "Skipping malformed history frame %zu (size %zu != %zu)", i, frame.size(), obs_dim_);
        for (size_t j = 0; j < obs_dim_ && idx < input.size(); ++j) {
          input[idx++] = current_frame[j];
        }
        continue;
      }
      for (size_t j = 0; j < obs_dim_ && idx < input.size(); ++j) {
        input[idx++] = frame[j];
      }
    }
  } else {
    // 历史帧足够：把历史帧（取最近的 take_from_history 帧）按照时间从旧到新添加
    size_t start_idx = (filled > take_from_history) ? (filled - take_from_history) : 0;
    for (size_t i = start_idx; i < filled; ++i) {
      const auto &frame = history_frames_[i];
      if (frame.size() != obs_dim_) {
        auto node = node_.lock();
        if (node) RCLCPP_WARN(node->get_logger(), "Skipping malformed history frame %zu (size %zu != %zu)", i, frame.size(), obs_dim_);
        for (size_t j = 0; j < obs_dim_ && idx < input.size(); ++j) {
          input[idx++] = current_frame[j];
        }
        continue;
      }
      for (size_t j = 0; j < obs_dim_ && idx < input.size(); ++j) {
        input[idx++] = frame[j];
      }
    }
  }

  // 把当前帧追加到输入（作为最后一帧），但不要立刻把模型输出写入历史；推理完成后再保存更新后的帧到 history_frames_
  for (size_t j = 0; j < obs_dim_ && idx < input.size(); ++j) {
    input[idx++] = current_frame[j];
  }

  // 返回当前构造的帧，供调用者在推理后填充模型输出并保存
  current_frame_out = std::move(current_frame);
  return input;
}

// 基于 costmap 计算前方 180° 的 20 个扇区最小距离（每扇区 9°），无 costmap/pose 时返回 0（表示最近，触发停止）
std::vector<float> RLController::computeObsFromCostmap(const geometry_msgs::msg::PoseStamped * pose)
{
  std::vector<float> obs(min_obs_dim_, 0.0f);

  // 如果没有 costmap 或 pose：把障碍距离拉到最近（0），让控制器停止
  if (!costmap_ || !pose) {
    last_obs_min_dist_ = 0.0;
    for (size_t i = 0; i < min_obs_dim_; ++i) obs[i] = 0.0f;
    return obs;
  }

  const double max_range = 10.0; // 最大探测范围（米）
  double cx = pose->pose.position.x;
  double cy = pose->pose.position.y;
  double yaw = yawFromQuat(pose->pose.orientation);
  double global_min = std::numeric_limits<double>::infinity();

  double map_res = costmap_->getResolution();
  double step = std::max(0.01, map_res * 0.5); // 步进距离，至少 1cm
  int max_steps = static_cast<int>(std::ceil(max_range / step));
  const double PI = std::acos(-1.0);
  // 前方 180°：从 -PI/2 到 +PI/2，分成 min_obs_dim_ 个扇区
  double start_angle = -PI / 2.0;
  double sector_width = PI / static_cast<double>(min_obs_dim_);

  // 对每个扇区，在扇区内投多条射线（rays_per_sector_ 条），取最小距离作为该扇区值
  for (size_t i = 0; i < min_obs_dim_; ++i) {
    double sector_center = start_angle + (static_cast<double>(i) + 0.5) * sector_width;
    double sector_min = max_range;
    // 若只取 1 条射线，则采样扇区中心线；否则在扇区内均匀分布射线
    for (size_t r = 0; r < std::max<size_t>(1, rays_per_sector_); ++r) {
      double offset = 0.0;
      if (rays_per_sector_ > 1) {
        offset = -sector_width / 2.0 + (static_cast<double>(r) * (sector_width / static_cast<double>(rays_per_sector_ - 1)));
      }
      double dir = yaw + sector_center + offset;
      double found_dist = max_range;
      for (int s = 1; s <= max_steps; ++s) {
        double d = s * step;
        if (d > max_range) break;
        double wx = cx + std::cos(dir) * d;
        double wy = cy + std::sin(dir) * d;
        unsigned int mx, my;
        if (!costmap_->worldToMap(wx, wy, mx, my)) {
          // 超出地图边界，认为该射线没有遇到障碍，保持 max_range 并结束该射线
          break;
        }
        // 防御性检查：有时 worldToMap 可能返回边界值但索引仍然越界，额外验证
        unsigned int size_x = costmap_->getSizeInCellsX();
        unsigned int size_y = costmap_->getSizeInCellsY();
        if (mx >= size_x || my >= size_y) {
          auto node = node_.lock();
          if (node) {
            RCLCPP_WARN(node->get_logger(), "Costmap index out of range: mx=%u my=%u size_x=%u size_y=%u, treating as free", mx, my, size_x, size_y);
          }
          // 认为射线未命中障碍，继续下一条射线
          continue;
        }
        unsigned char c = costmap_->getCost(mx, my);
        if (c >= nav2_costmap_2d::LETHAL_OBSTACLE || c > 0) {
          found_dist = d;
          break;
        }
      }
      if (found_dist < sector_min) sector_min = found_dist;
      if (sector_min < global_min) global_min = sector_min;
    }
    obs[i] = static_cast<float>(sector_min);
  }

  last_obs_min_dist_ = (std::isfinite(global_min) ? global_min : max_range);
  return obs;
}

// 使用 ONNX Runtime 运行前向推理，输入为 [1, model_input_size_] 的一维向量，返回模型输出向量
std::vector<float> RLController::runModel(const std::vector<float> & input)
{
  std::vector<float> result;
  try {
    // Lazy initialize ONNX env/session if not already created
    if (!ort_session_) {
      std::lock_guard<std::mutex> lock(ort_mutex_);
      if (ort_failed_) return result;

      // Ensure Ort::Env exists (may have been created in configure)
      if (!ort_env_) {
        try {
          ort_env_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, plugin_name_.c_str()));
          auto node = node_.lock();
          if (node) RCLCPP_INFO(node->get_logger(), "Ort::Env initialized lazily for %s", plugin_name_.c_str());
        } catch (const std::exception & e) {
          auto node = node_.lock();
          if (node) RCLCPP_ERROR(node->get_logger(), "Failed to initialize Ort::Env lazily: %s", e.what());
          ort_failed_ = true;
          return result;
        } catch (...) {
          auto node = node_.lock();
          if (node) RCLCPP_ERROR(node->get_logger(), "Unknown failure initializing Ort::Env lazily");
          ort_failed_ = true;
          return result;
        }
      }

      if (!ort_session_ && !ort_failed_) {
        try {
          // Ort::Session expects a reference to Ort::Env
          ort_session_.reset(new Ort::Session(*ort_env_, model_path_.c_str(), session_options_));
          auto node = node_.lock();
          if (node) RCLCPP_INFO(node->get_logger(), "Loaded ONNX model lazily: %s", model_path_.c_str());
        } catch (const std::exception & e) {
          auto node = node_.lock();
          if (node) RCLCPP_ERROR(node->get_logger(), "Failed to load ONNX model lazily: %s", e.what());
          ort_failed_ = true;
          return result;
        }
      }
    }

    if (!ort_session_) return result;

    Ort::AllocatorWithDefaultOptions allocator;
    // Input name(s) - use AllocatedStringPtr to keep storage alive while calling Run
    size_t num_input_nodes = ort_session_->GetInputCount();
    std::vector<const char*> input_node_names;
    input_node_names.reserve(num_input_nodes);
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    input_name_ptrs.reserve(num_input_nodes);
    for (size_t i = 0; i < num_input_nodes; ++i) {
      auto name_ptr = ort_session_->GetInputNameAllocated(i, allocator);
      input_node_names.push_back(name_ptr.get());
      input_name_ptrs.push_back(std::move(name_ptr));
    }

    // Output names
    size_t num_output_nodes = ort_session_->GetOutputCount();
    std::vector<const char*> output_node_names;
    output_node_names.reserve(num_output_nodes);
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    output_name_ptrs.reserve(num_output_nodes);
    for (size_t i = 0; i < num_output_nodes; ++i) {
      auto name_ptr = ort_session_->GetOutputNameAllocated(i, allocator);
      output_node_names.push_back(name_ptr.get());
      output_name_ptrs.push_back(std::move(name_ptr));
    }

    // Prepare input tensor with shape [1, model_input_size_]
    std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(model_input_size_)};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(input.data()), input.size(), input_shape.data(), input_shape.size());

    // Run
    auto output_tensors = ort_session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), output_node_names.size());

    // Assume first output is the action vector
    if (output_tensors.size() > 0) {
      float* out_data = output_tensors[0].GetTensorMutableData<float>();
      auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
      size_t out_size = 1;
      for (auto d : out_shape) out_size *= d;
      result.resize(out_size);
      for (size_t i = 0; i < out_size; ++i) result[i] = out_data[i];
    }
    return result;
  } catch (const std::exception & e) {
    auto node = node_.lock();
    if (node) RCLCPP_ERROR(node->get_logger(), "runModel exception: %s", e.what());
    ort_failed_ = true;
    return result;
  } catch (...) {
    auto node = node_.lock();
    if (node) RCLCPP_ERROR(node->get_logger(), "runModel unknown exception");
    ort_failed_ = true;
    return result;
  }
} 

// 计算到目标点的sin、cos、distance
std::tuple<double,double,double> RLController::computeTargetFromPlan(const geometry_msgs::msg::PoseStamped & current_pose)
{
  std::lock_guard<std::mutex> lock(plan_mutex_);
  //如果有全局路径才计算，否则返回全零（但全零不符合到达目标的条件）
  if (!have_plan_ || latest_plan_.poses.empty()) {
    return {0.0, 0.0, 0.0};
  }

  // 找到距离当前位姿最近的路径点的索引（还有一种写法是到路径中索引最近的点，但这里先直接找最近点）
  double best_dist = std::numeric_limits<double>::infinity();
  size_t best_idx = 0;
  double cx = current_pose.pose.position.x;
  double cy = current_pose.pose.position.y;
  for (size_t i = 0; i < latest_plan_.poses.size(); ++i) {
    double dx = latest_plan_.poses[i].pose.position.x - cx;
    double dy = latest_plan_.poses[i].pose.position.y - cy;
    double d = std::hypot(dx, dy);
    if (d < best_dist) { best_dist = d; best_idx = i; }
  }

  // 目标点选择为最近点的下一个点（若存在），否则取最近点
  size_t target_idx = (best_idx + 1 < latest_plan_.poses.size()) ? best_idx + 1 : best_idx;
  const auto &tp = latest_plan_.poses[target_idx].pose;
  double dx = tp.position.x - cx;
  double dy = tp.position.y - cy;
  double dist = std::hypot(dx, dy);
  double yaw = yawFromQuat(current_pose.pose.orientation);
  double angle_to_target = std::atan2(dy, dx) - yaw;
  const double PI = std::acos(-1.0);
  while (angle_to_target > PI) angle_to_target -= 2.0 * PI;
  while (angle_to_target < -PI) angle_to_target += 2.0 * PI;
  double tcos = std::cos(angle_to_target);
  double tsin = std::sin(angle_to_target);
  return {tcos, tsin, dist};
}

// 辅助函数：从四元数计算 yaw（弧度）
double RLController::yawFromQuat(const geometry_msgs::msg::Quaternion & q)
{
  double siny = 2.0 * (q.w * q.z + q.x * q.y);
  double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny, cosy);
}

// 路径稀疏化：每distance_threshold米保留一个路径点
nav_msgs::msg::Path RLController::sparsePath(const nav_msgs::msg::Path & path, double distance_threshold)
{
  nav_msgs::msg::Path sparse_path;
  sparse_path.header = path.header;
  
  if (path.poses.empty()) {
    return sparse_path;
  }
  
  // 总是保留第一个点
  sparse_path.poses.push_back(path.poses[0]);
  
  if (path.poses.size() == 1) {
    return sparse_path;
  }
  
  // 从第一个点开始，累积距离，当距离超过阈值时保留该点
  size_t last_kept_idx = 0;
  for (size_t i = 1; i < path.poses.size(); ++i) {
    const auto &last_pose = path.poses[last_kept_idx].pose;
    const auto &current_pose = path.poses[i].pose;
    
    double dx = current_pose.position.x - last_pose.position.x;
    double dy = current_pose.position.y - last_pose.position.y;
    double distance = std::hypot(dx, dy);
    
    // 如果距离超过阈值，保留当前点
    if (distance >= distance_threshold) {
      sparse_path.poses.push_back(path.poses[i]);
      last_kept_idx = i;
    }
  }
  
  // 总是保留最后一个点（即使距离不够）
  if (sparse_path.poses.empty() || 
    sparse_path.poses.back().pose.position.x != path.poses.back().pose.position.x ||
    sparse_path.poses.back().pose.position.y != path.poses.back().pose.position.y) {
    sparse_path.poses.push_back(path.poses.back());
  }
  
  return sparse_path;
}

// 辅助函数：将一帧观测保存到文本文件
void RLController::saveObservationToFile(const std::vector<float>& obs) {
  // 确保有足够的维度
  if (obs.size() < 25) return;
  
  // 打开文件（追加模式）
  std::ofstream outfile("/home/unitree/nav2_gps/nav2_rl_controller/observations.txt", std::ios_base::app);
  
  if (!outfile.is_open()) {
    // RCLCPP_WARN(this->get_logger(), "无法打开文件保存观测数据");
    return;
  }
  
  // 写入时间戳（可选）
  auto now = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);
  outfile << "Time: " << std::ctime(&now_time);
  
  // 第一行：前20个值（扇区化距离）
  outfile << "扇区观测: ";
  for (size_t i = 0; i < 20; ++i) {
    outfile << obs[i];
    if (i < 19) outfile << ", ";
  }
  outfile << std::endl;
  
  // 第二行：中间3个值（目标信息）
  outfile << "目标信息: ";
  outfile << obs[20] << ", " << obs[21] << ", " << obs[22];
  outfile << std::endl;
  
  // 第三行：最后2个值（动作信息）
  outfile << "动作信息: ";
  outfile << obs[23] << ", " << obs[24];
  outfile << std::endl;
  
  // 添加分隔线
  outfile << "----------------------------------------" << std::endl;
  
  outfile.close();
}

}  // namespace nav2_rl_controller

// 注册到nav2插件系统
PLUGINLIB_EXPORT_CLASS(nav2_rl_controller::RLController, nav2_core::Controller);
