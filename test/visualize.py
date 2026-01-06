import numpy as np
import matplotlib.pyplot as plt


def draw_lidar_sectors(
        sector_distances,
        target_info,
        save_path="costmap.png"
):
    """
    sector_distances: list or np.array, length=20
    clockwise: True=顺时针绘制, False=逆时针绘制
    """

    target_cos = target_info[0]
    target_sin = target_info[1]
    target_distance = target_info[2]

    num_sectors = 20
    sector_angle = np.deg2rad(180 / num_sectors)  # 9°

    # 扇区中心角（-90° 到 +90°）
    angles = np.linspace(
        0 + sector_angle / 2,
        np.pi - sector_angle / 2,
        num_sectors
    )

    # ros雷达逆时针给数据，所以直接按0-180度画就行
    # angles = angles[::-1]

    fig, ax = plt.subplots(figsize=(6, 6))

    # 画每个扇区（用线段表示最近障碍）
    for d, ang in zip(sector_distances, angles):
        x = d * np.cos(ang)
        y = d * np.sin(ang)

        # 坐标变换
        plot_x = x
        plot_y = y

        ax.plot([0, plot_x], [0, plot_y], color="tab:blue", linewidth=2)

    # 画机器人
    ax.scatter(0, 0, color="black", s=50, zorder=5)

    # 目标点
    target_x = target_distance * target_cos
    target_y = target_distance * target_sin

    plot_tx = -target_y
    plot_ty = target_x

    ax.scatter(plot_tx, plot_ty, color="red", s=40, zorder=6)

    # 外观设置
    ax.set_aspect("equal")
    max_range = max(sector_distances) + 1.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    ax.set_xlabel("y (horizon)")
    ax.set_ylabel("x (forward)")

    ax.grid(True)

    # 保存，不显示
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    sector_obs = [10, 3.6, 3.2, 3.2, 3.25, 3.65, 4.4, 3.4, 3.1, 3.1, 2.85, 1.65, 1.2, 1.05, 0.95, 0.95, 0.9, 0.95, 0.95, 1.05]

    target_info = [-0.288599, -0.95745, 2.59999]

    draw_lidar_sectors(
        sector_distances=sector_obs,
        target_info=target_info,
        save_path="costmap.png"
    )
