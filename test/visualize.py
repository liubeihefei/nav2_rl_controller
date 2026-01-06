import numpy as np
import matplotlib.pyplot as plt


def draw_lidar_sectors(
        sector_distances,
        target_cos,
        target_sin,
        target_distance,
        clockwise=True,
        save_path="lidar_sector_plot.png"
):
    """
    sector_distances: list or np.array, length=20
    clockwise: True=顺时针绘制, False=逆时针绘制
    """

    assert len(sector_distances) == 20, "必须是20个扇区距离"

    num_sectors = 20
    sector_angle = np.deg2rad(180 / num_sectors)  # 9°

    # 扇区中心角（-90° 到 +90°）
    angles = np.linspace(
        -np.pi / 2 + sector_angle / 2,
        +np.pi / 2 - sector_angle / 2,
        num_sectors
    )

    if clockwise:
        angles = angles[::-1]

    fig, ax = plt.subplots(figsize=(6, 6))

    # 画每个扇区（用线段表示最近障碍）
    for d, ang in zip(sector_distances, angles):
        x = d * np.cos(ang)
        y = d * np.sin(ang)

        # 坐标变换：x向前画成向上
        plot_x = y
        plot_y = x

        ax.plot([0, plot_x], [0, plot_y], color="tab:blue", linewidth=2)

    # 画机器人
    ax.scatter(0, 0, color="black", s=50, zorder=5)
    # ax.text(0, 0, "Robot", fontsize=10, ha="right", va="bottom")

    # 目标点
    target_x = target_distance * target_cos
    target_y = target_distance * target_sin

    plot_tx = target_y
    plot_ty = target_x

    ax.scatter(plot_tx, plot_ty, color="red", s=40, zorder=6)
    # ax.text(plot_tx, plot_ty, "Target", color="red", fontsize=10)

    # 外观设置
    ax.set_aspect("equal")
    max_range = max(sector_distances) + 1.0
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(0, max_range)

    ax.set_xlabel("y (horizon)")
    ax.set_ylabel("x (forward)")
    # ax.set_title("Lidar Sector Observation (Robot Facing Up)")

    ax.grid(True)

    # 保存，不显示
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    sector_obs = [
        4.75, 3.7, 3.55, 3.55, 3.85,
        4.75, 5.4, 10, 10, 10,
        10, 2.8, 1.6, 1.25, 1.15,
        1.1, 1.05, 1.05, 1.1, 1.15
    ]

    target_cos = 0.234133
    target_sin = -0.972205
    target_distance = 2.10805

    draw_lidar_sectors(
        sector_distances=sector_obs,
        target_cos=target_cos,
        target_sin=target_sin,
        target_distance=target_distance,
        clockwise=True,  # ← 改这里切换顺 / 逆时针
        save_path="lidar_view.png"
    )
