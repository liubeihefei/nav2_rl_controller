import re
import ast
import numpy as np
import cv2
import matplotlib.pyplot as plt

LOG_FILE = 'D://Projects//nav2_rl_controller//logs//logs666//observations.txt'
VIDEO_OUT = "result.mp4"

history_obstacles = []
start_time = None


# -----------------------------
# 四元数 -> yaw
# -----------------------------
def quat_to_yaw(q):
    x, y, z, w = q
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny, cosy)


# -----------------------------
# odom -> map（只平移）
# -----------------------------
def odom_to_map(px, py, tf_trans):

    tx, ty, _ = tf_trans

    mx = px + tx
    my = py + ty

    return mx, my


# -----------------------------
# 机器人坐标 -> 世界坐标
# -----------------------------
def robot_to_world(px, py, yaw, x, y):

    mx = px + np.cos(yaw) * x - np.sin(yaw) * y
    my = py + np.sin(yaw) * x + np.cos(yaw) * y

    return mx, my


# -----------------------------
# 解析日志
# -----------------------------
def parse_log(file):

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    blocks = text.split("----------------------------------------")

    frames = []

    for b in blocks:

        if "Timestamp" not in b:
            continue

        ts = int(re.search(r"Timestamp:\s*(\d+)", b).group(1))

        lidar = list(map(float,
            re.search(r"扇区观测:\s*(.*)", b).group(1).split(",")
        ))

        target = list(map(float,
            re.search(r"目标信息:\s*(.*)", b).group(1).split(",")
        ))

        action = list(map(float,
            re.search(r"动作信息:\s*(.*)", b).group(1).split(",")
        ))

        path = ast.literal_eval(
            re.search(r"路径点:\s*(\[.*\])", b).group(1)
        )

        pos = ast.literal_eval(
            re.search(r"当前位置：(\[.*\])", b).group(1)
        )

        quat = ast.literal_eval(
            re.search(r"当前朝向：(\[.*\])", b).group(1)
        )

        tf_trans = ast.literal_eval(
            re.search(r"map->odom translation:\s*(\[.*\])", b).group(1)
        )

        frames.append({
            "timestamp": ts,
            "lidar": lidar,
            "target": target,
            "action": action,
            "path": path,
            "pos": pos,
            "quat": quat,
            "tf_trans": tf_trans
        })

    return frames


# -----------------------------
# 雷达点
# -----------------------------
def lidar_points(frame):

    px_o, py_o = frame["pos"]
    # px, py = odom_to_map(px_o, py_o, frame["tf_trans"])
    px, py = px_o, py_o

    yaw = quat_to_yaw(frame["quat"])

    lidar = frame["lidar"]

    pts = []

    start = -np.pi / 2
    step = np.pi / 20

    for i, d in enumerate(lidar):

        if d >= 10:
            continue

        ang = start + i * step

        rx = d * np.cos(ang)
        ry = d * np.sin(ang)

        mx, my = robot_to_world(px, py, yaw, rx, ry)

        pts.append((mx, my))

    return np.array(pts)


# -----------------------------
# 目标点
# -----------------------------
def target_point(frame):

    dist, c, s = frame["target"]

    rx = dist * c
    ry = dist * s

    px_o, py_o = frame["pos"]
    # px, py = odom_to_map(px_o, py_o, frame["tf_trans"])

    yaw = quat_to_yaw(frame["quat"])

    # return robot_to_world(px, py, yaw, rx, ry)
    return robot_to_world(px_o, py_o, yaw, rx, ry)


# -----------------------------
# 画一帧
# -----------------------------
def draw_frame(frame):

    global history_obstacles
    global start_time

    px_o, py_o = frame["pos"]
    # px, py = odom_to_map(px_o, py_o, frame["tf_trans"])
    px, py = px_o, py_o

    yaw = quat_to_yaw(frame["quat"])

    lidar = lidar_points(frame)

    if len(lidar) > 0:
        history_obstacles.extend(lidar.tolist())

    tx, ty = target_point(frame)

    path = np.array(frame["path"])

    pts_list = [path, [[px, py]], [[tx, ty]]]

    if len(lidar) > 0:
        pts_list.append(lidar)

    if len(history_obstacles) > 0:
        pts_list.append(np.array(history_obstacles))

    pts = np.vstack(pts_list)

    xmin, ymin = pts.min(axis=0) - 0.5
    xmax, ymax = pts.max(axis=0) + 0.5

    fig, ax = plt.subplots(figsize=(6,6))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # 历史障碍物
    if len(history_obstacles) > 0:
        hist = np.array(history_obstacles)
        ax.scatter(hist[:,0], hist[:,1], c="gray", s=1)

    # 当前雷达
    if len(lidar) > 0:

        ax.scatter(lidar[:,0], lidar[:,1], c="blue", s=10)

        for p in lidar:
            ax.plot([px, p[0]], [py, p[1]], c="blue", linewidth=0.5)

    # 目标
    ax.scatter(tx, ty, c="red", s=50)

    # 路径
    if len(path) > 0:
        ax.scatter(path[:,0], path[:,1], c="green", s=20)

    # 机器人
    ax.scatter(px, py, c="black", s=80)

    # 朝向
    dx = np.cos(yaw)
    dy = np.sin(yaw)

    ax.arrow(px, py, dx, dy, width=0.02)

    # 时间
    elapsed = (frame["timestamp"] - start_time) / 1000.0

    text = f"timestamp: {frame['timestamp']}\n"
    text += f"t: {elapsed:.2f}s\n"
    text += f"v: {frame['action'][0]:.2f}\n"
    text += f"w: {frame['action'][1]:.2f}"

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment='top'
    )

    ax.set_aspect("equal")

    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3].copy()

    plt.close()

    return img


# -----------------------------
# 主程序
# -----------------------------
frames = parse_log(LOG_FILE)

start_time = frames[0]["timestamp"]

video = None

for f in frames:

    img = draw_frame(f)

    if video is None:

        h, w, _ = img.shape

        video = cv2.VideoWriter(
            VIDEO_OUT,
            cv2.VideoWriter_fourcc(*'mp4v'),
            10,
            (w, h)
        )

    video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

video.release()