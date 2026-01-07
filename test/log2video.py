import cv2
import numpy as np
import re
import os


def parse_log_file(log_file, max_frames=100):
    """解析日志文件"""
    frames_data = []

    try:
        # 尝试不同编码
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(log_file, 'r', encoding='gbk') as f:
                content = f.read()

        # 分割日志为块
        blocks = content.split('----------------------------------------\n')

        for block in blocks:
            if len(frames_data) >= max_frames:
                break

            lines = block.strip().split('\n')
            if len(lines) < 4:
                continue

            frame_data = {}

            for line in lines:
                if line.startswith('Time:'):
                    frame_data['time'] = line[5:].strip()
                elif '扇区观测' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        # 提取数字
                        nums = re.findall(r'[-+]?\d*\.\d+|\d+', parts[1])
                        if len(nums) >= 20:
                            frame_data['sector_distances'] = [float(x) for x in nums[:20]]
                elif '目标信息' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        nums = re.findall(r'[-+]?\d*\.\d+|\d+', parts[1])
                        if len(nums) >= 3:
                            frame_data['target_info'] = [float(x) for x in nums[:3]]
                elif '动作信息' in line:
                    parts = line.split(':')
                    if len(parts) > 1:
                        action_str = parts[1].strip()
                        # 提取线速度和角速度
                        action_nums = re.findall(r'[-+]?\d*\.\d+|\d+', action_str)
                        if len(action_nums) >= 2:
                            frame_data['linear_vel'] = float(action_nums[0])
                            frame_data['angular_vel'] = float(action_nums[1])
                        else:
                            frame_data['linear_vel'] = 0.0
                            frame_data['angular_vel'] = 0.0

            # 检查数据完整性
            if ('sector_distances' in frame_data and
                    'target_info' in frame_data and
                    len(frame_data['sector_distances']) == 20):
                frames_data.append(frame_data)

    except Exception as e:
        print(f"解析错误: {e}")
        # 创建示例数据
        frames_data = create_sample_frames(min(30, max_frames))

    print(f"解析了 {len(frames_data)} 帧数据")
    return frames_data


def create_sample_frames(num_frames=30):
    """创建示例帧数据"""
    frames_data = []
    for i in range(num_frames):
        # 障碍物数据 - 模拟前方有障碍物
        sectors = [10.0] * 20
        sectors[8:12] = [3.0, 2.0, 1.5, 2.0]  # 前方中间有障碍

        # 目标点 - 模拟移动
        angle = i * 0.1
        target_cos = np.cos(angle)
        target_sin = np.sin(angle)
        distance = max(1.5, 5.0 - i * 0.1)

        # 动作 - 模拟控制
        linear_vel = max(0.1, 0.5 - i * 0.01)
        angular_vel = 0.1 if i % 10 < 5 else -0.1

        frame_data = {
            'time': f'Tue Jan  6 14:49:{i:02d} 2026',
            'sector_distances': sectors,
            'target_info': [target_cos, target_sin, distance],
            'linear_vel': linear_vel,
            'angular_vel': angular_vel
        }
        frames_data.append(frame_data)

    return frames_data


def draw_costmap_frame(frame_data, frame_index, total_frames, frame_size=600):
    """
    绘制简洁的costmap帧
    """
    # 创建深色背景
    frame_img = np.ones((frame_size, frame_size, 3), dtype=np.uint8) * 30  # 深灰背景

    # 获取数据
    sector_distances = frame_data.get('sector_distances', [10.0] * 20)
    target_info = frame_data.get('target_info', [0.0, 0.0, 10.0])
    target_cos, target_sin, target_distance = target_info
    linear_vel = frame_data.get('linear_vel', 0.0)
    angular_vel = frame_data.get('angular_vel', 0.0)

    # 确保扇区数据正确
    if len(sector_distances) != 20:
        sector_distances = [10.0] * 20

    # 中心点和缩放
    center_x, center_y = frame_size // 2, frame_size // 2
    scale_factor = frame_size / 22  # 10米对应约一半的半径

    # 绘制半圆参考线
    for radius_m in [2, 5, 10]:
        radius_px = int(radius_m * scale_factor)
        # 只画前面180度的半圆
        cv2.ellipse(frame_img, (center_x, center_y), (radius_px, radius_px),
                    0, 0, 360, (80, 80, 80), 1)

        # 标注距离（放在右侧）
        cv2.putText(frame_img, f"{radius_m}m",
                    (center_x + radius_px + 5, center_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # 绘制方向指示
    cv2.putText(frame_img, "+90", (center_x - 150, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(frame_img, "0", (center_x - 5, center_y - 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(frame_img, "-90", (center_x + 140, center_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    # 绘制20条扇区线（前面180度）
    num_sectors = 20
    angles = np.linspace(np.pi / 20, np.pi - np.pi / 20, num_sectors)

    for i, (d, ang) in enumerate(zip(sector_distances, angles)):
        # 限制显示距离
        display_d = min(d, 10.0)

        # 计算端点
        x = display_d * np.cos(ang) * scale_factor
        y = -display_d * np.sin(ang) * scale_factor  # y轴向下

        end_x = int(center_x + x)
        end_y = int(center_y + y)

        # 根据距离设置颜色和线宽
        if d < 1.0:  # 非常近，红色
            color = (0, 0, 255)
            thickness = 1
        elif d < 2.0:  # 近，橙色
            color = (0, 165, 255)
            thickness = 1
        elif d < 5.0:  # 中等，黄色
            color = (0, 255, 255)
            thickness = 1
        else:  # 远，绿色
            color = (0, 255, 0)
            thickness = 1

        # 绘制扇区线
        cv2.line(frame_img, (center_x, center_y), (end_x, end_y), color, thickness)

        # 在末端绘制小点
        cv2.circle(frame_img, (end_x, end_y), 3, color, -1)

    # 绘制机器人位置（中心点）
    cv2.circle(frame_img, (center_x, center_y), 8, (255, 255, 255), -1)
    cv2.circle(frame_img, (center_x, center_y), 10, (200, 200, 200), 2)

    # 绘制目标点
    target_x = target_distance * target_cos * scale_factor
    target_y = -target_distance * target_sin * scale_factor

    target_pixel_x = int(center_x + target_x)
    target_pixel_y = int(center_y + target_y)

    # 确保目标点在图像内
    target_pixel_x = max(0, min(frame_size - 1, target_pixel_x))
    target_pixel_y = max(0, min(frame_size - 1, target_pixel_y))

    # 绘制目标点和连线
    cv2.line(frame_img, (center_x, center_y), (target_pixel_x, target_pixel_y),
             (255, 100, 100), 2, cv2.LINE_AA)
    cv2.circle(frame_img, (target_pixel_x, target_pixel_y), 10, (255, 50, 50), -1)
    cv2.circle(frame_img, (target_pixel_x, target_pixel_y), 12, (255, 100, 100), 2)

    # ========== 信息显示区域（右侧统一放置） ==========
    info_start_x = 20
    info_y = 50
    line_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX

    # # 背景框
    # cv2.rectangle(frame_img,
    #               (info_start_x - 10, 20),
    #               (frame_size - 20, 180),
    #               (40, 40, 40), -1)
    # cv2.rectangle(frame_img,
    #               (info_start_x - 10, 20),
    #               (frame_size - 20, 180),
    #               (80, 80, 80), 1)

    # 帧信息
    frame_text = f"Frame: {frame_index + 1}/{total_frames}"
    cv2.putText(frame_img, frame_text,
                (info_start_x, info_y),
                font, 0.7, (200, 200, 200), 2)

    # 目标信息
    target_text = f"Target: {target_distance:.2f}m"
    cv2.putText(frame_img, target_text,
                (info_start_x, info_y + line_height),
                font, 0.6, (255, 100, 100), 2)

    # 最近障碍物信息
    min_dist = min(sector_distances)
    min_idx = sector_distances.index(min_dist)
    obstacle_text = f"Min Dist: {min_dist:.2f}m"
    cv2.putText(frame_img, obstacle_text,
                (info_start_x, info_y + line_height * 2),
                font, 0.6,
                (255, 100, 100) if min_dist < 1.0 else (100, 255, 100), 2)

    # 控制指令（线速度和角速度）
    vel_text = f"Linear: {linear_vel:.3f} m/s"
    cv2.putText(frame_img, vel_text,
                (info_start_x, info_y + line_height * 3),
                font, 0.6, (100, 200, 255), 2)

    angular_text = f"Angular: {angular_vel:.3f} rad/s"
    cv2.putText(frame_img, angular_text,
                (info_start_x, info_y + line_height * 4),
                font, 0.6, (100, 200, 255), 2)

    # 控制方向指示（如果角速度不为0）
    if abs(angular_vel) > 0.01:
        direction = "LEFT" if angular_vel > 0 else "RIGHT"
        dir_color = (255, 150, 0) if angular_vel > 0 else (0, 150, 255)
        dir_text = f"Turning: {direction}"
        cv2.putText(frame_img, dir_text,
                    (info_start_x, info_y + line_height * 5),
                    font, 0.6, dir_color, 2)

    return frame_img


def create_costmap_video(log_file, output_video="../logs/costmap_video.mp4", fps=10, max_frames=100):
    """创建costmap视频"""
    print(f"处理文件: {log_file}")

    # 检查文件
    if not os.path.exists(log_file):
        print(f"文件不存在，创建示例数据...")
        frames_data = create_sample_frames(max_frames)
    else:
        frames_data = parse_log_file(log_file, max_frames)

    if not frames_data:
        print("无有效数据，使用示例数据")
        frames_data = create_sample_frames(min(30, max_frames))

    total_frames = len(frames_data)
    print(f"生成 {total_frames} 帧视频，帧率 {fps}fps")

    # 创建视频
    frame_size = 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_size, frame_size))

    if not video_writer.isOpened():
        print(f"无法创建视频文件")
        return

    # 生成帧
    for i, frame_data in enumerate(frames_data):
        frame_img = draw_costmap_frame(frame_data, i, total_frames, frame_size)
        video_writer.write(frame_img)

        # 进度显示
        if (i + 1) % 10 == 0 or i == 0 or i == total_frames - 1:
            progress = (i + 1) / total_frames * 100
            print(f"进度: {i + 1}/{total_frames} ({progress:.1f}%)")

    video_writer.release()
    print(f"视频已保存: {output_video}")

    # 显示第一帧作为预览
    preview_img = draw_costmap_frame(frames_data[0], 0, total_frames, frame_size)
    cv2.imshow("Costmap Preview (Press any key)", preview_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """主函数 - 简洁版本"""
    print("=== Costmap 可视化工具 ===\n")

    # 简单参数设置
    log_file = input("日志文件路径 [默认: observations.txt]: ").strip()
    if not log_file:
        log_file = "../logs/observations.txt"

    output_video = input("输出视频 [默认: costmap.mp4]: ").strip()
    if not output_video:
        output_video = "../logs/costmap.mp4"

    # 自动处理
    create_costmap_video(log_file, output_video, fps=10, max_frames=10000)


if __name__ == "__main__":
    main()