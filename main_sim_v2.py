"""
多无人机覆盖控制仿真主程序
简化版：只输出轨迹图和覆盖代价曲线
"""
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from typing import Tuple
import os
import sys
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.sensitivity_field_v1 import DynamicSensitivityField, MotionType
from src.prediction.gp_predictor import SpatioTemporalGP
from src.coverage.voronoi import WeightedVoronoi
from src.coverage.lloyd_controller import LloydController
from src.safety.cbf import DistributedCBF
from src.agents.uav import UAVSwarm


class CoverageSimulator:
    """覆盖控制仿真器 - 简化版"""

    def __init__(self, num_agents: int = 5,
                 domain: Tuple = (0, 100, 0, 100),
                 total_time: float = 30.0,
                 dt: float = 0.1,
                 field_preset: str = "mixed",
                 use_cbf: bool = True,
                 use_gp_prediction: bool = True):
        """
        Args:
            num_agents: 无人机数量
            domain: 工作区域
            total_time: 仿真总时长
            dt: 时间步长
            field_preset: 敏感度场预设 ("static", "linear", "circular", "mixed")
            use_cbf: 是否使用CBF安全约束
            use_gp_prediction: 是否使用GP预测
        """
        self.num_agents = num_agents
        self.domain = domain
        self.total_time = total_time
        self.dt = dt
        self.field_preset = field_preset
        self.use_cbf = use_cbf
        self.use_gp_prediction = use_gp_prediction

        # 初始化组件
        self._init_components()

        # 仿真状态
        self.time = 0.0
        self.step_count = 0

        # 历史数据
        self.history = {
            'time': [],
            'positions': [],  # 每步的位置
            'coverage_cost': [],  # 覆盖代价
            'weighted_coverage': [],  # 加权覆盖率
            'field_snapshots': [],  # 敏感度场快照（用于动画）
            'hotspot_positions': []  # 热点位置
        }

        # 输出目录
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def _init_components(self):
        """初始化仿真组件"""
        # 1. 动态敏感度场
        self.field = DynamicSensitivityField(self.domain, resolution=50)
        self.field.add_preset_hotspots(self.field_preset, seed=42)

        # 2. 无人机集群
        self.swarm = UAVSwarm(
            self.num_agents,
            self.domain,
            max_velocity=5.0,
            sensing_radius=15.0
        )

        # 3. 覆盖控制器
        self.controller = LloydController(
            self.domain,
            gain=1.5,
            max_velocity=5.0,
            resolution=50
        )

        # 4. Voronoi计算
        self.voronoi = WeightedVoronoi(self.domain, resolution=50)

        # 5. CBF安全滤波
        if self.use_cbf:
            self.cbf = DistributedCBF(
                safe_distance=3.0,
                gamma=1.0,
                domain=self.domain,
                communication_radius=30.0
            )

        # 6. GP预测器
        if self.use_gp_prediction:
            self.gp = SpatioTemporalGP(
                length_scale_space=15.0,
                length_scale_time=10.0,
                noise_variance=0.1
            )
            self.gp_trained = False
            self.prediction_horizon = 10

    def step(self):
        """执行一步仿真"""
        # 1. 更新敏感度场
        self.field.update(self.dt)
        current_field = self.field.get_field_grid()

        # 2. 感知（用于GP训练）
        if self.use_gp_prediction:
            self.swarm.sense_all(self.field, self.time)
            if self.step_count % 5 == 0:
                self._update_gp()

        # 3. 获取预测场
        if self.use_gp_prediction and self.gp_trained:
            predicted_field = self._predict_future_field()
        else:
            predicted_field = current_field

        # 4. 计算控制
        positions = self.swarm.get_positions()
        controls = self.controller.compute_predictive_control(
            positions, current_field, predicted_field, alpha=0.6
        )

        # 5. CBF安全滤波
        if self.use_cbf:
            controls = self.cbf.filter_control(positions, controls)

        # 6. 更新无人机
        self.swarm.update_all(controls, self.dt, use_velocity=True)

        # 7. 记录数据
        self._record_step(current_field)

        # 8. 更新时间
        self.time += self.dt
        self.step_count += 1

    def _update_gp(self):
        """更新GP模型"""
        X, y = self.swarm.get_all_sensed_data(time_window=20.0, current_time = self.time)
        if len(X) >= 20:
            try:
                self.gp.fit(X, y)
                self.gp_trained = True
            except:
                pass

    def _predict_future_field(self) -> np.ndarray:
        """预测未来敏感度场"""
        future_time = self.time + self.prediction_horizon * self.dt
        x_grid = np.linspace(self.domain[0], self.domain[1], 50)
        y_grid = np.linspace(self.domain[2], self.domain[3], 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        X_pred = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, future_time)])
        try:
            mean, _ = self.gp.predict(X_pred)
            return mean.reshape(50, 50)
        except:
            return self.field.get_field_grid()

    def _record_step(self, field: np.ndarray):
        """记录一步数据"""
        positions = self.swarm.get_positions()

        # 覆盖代价
        cost = self.voronoi.compute_coverage_cost(positions, field)

        # 加权覆盖率（简化计算）
        weighted_cov = self._compute_weighted_coverage(positions, field)

        self.history['time'].append(self.time)
        self.history['positions'].append(positions.copy())
        self.history['coverage_cost'].append(cost)
        self.history['weighted_coverage'].append(weighted_cov)

        # 每2步保存快照（用于动画）
        if self.step_count % 2 == 0:
            self.history['field_snapshots'].append(field.copy())
            self.history['hotspot_positions'].append(self.field.get_hotspot_positions().copy())

    def _compute_weighted_coverage(self, positions: np.ndarray, field: np.ndarray) -> float:
        """
        计算加权覆盖率
        定义：被无人机感知范围覆盖的敏感度占总敏感度的比例
        """
        sensing_radius = 15.0

        # 创建覆盖掩码
        x_grid = np.linspace(self.domain[0], self.domain[1], 50)
        y_grid = np.linspace(self.domain[2], self.domain[3], 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        # 计算每个网格点是否被覆盖
        covered = np.zeros(len(grid_points), dtype=bool)
        for pos in positions:
            distances = np.linalg.norm(grid_points - pos, axis=1)
            covered |= (distances <= sensing_radius)

        # 计算加权覆盖率
        field_flat = field.ravel()
        total_sensitivity = np.sum(field_flat)
        covered_sensitivity = np.sum(field_flat[covered])

        return covered_sensitivity / total_sensitivity if total_sensitivity > 0 else 0

    def run(self, verbose: bool = True):
        """运行仿真"""
        num_steps = int(self.total_time / self.dt)

        if verbose:
            print(f"Running simulation:  {num_steps} steps, {self.num_agents} UAVs")
            print(f"Field preset: {self.field_preset}")

        for i in range(num_steps):
            self.step()
            if verbose and i % 50 == 0:
                cost = self.history['coverage_cost'][-1]
                cov = self.history['weighted_coverage'][-1]
                print(f"  Step {i}/{num_steps}, Cost: {cost:.0f}, Coverage: {cov:.2%}")

        if verbose:
            print("Simulation complete!")

        return self.history

    # ==================== 可视化方法 ====================

    def plot_trajectory_frame(self, frame_idx: int = -1,
                              save_path: str = None,
                              show_centroids: bool = True):
        """
        绘制单帧轨迹图

        Args:
            frame_idx: 帧索引，-1表示最后一帧
            save_path: 保存路径
            show_centroids: 是否显示Voronoi质心
        """
        if not self.history['positions']:
            print("No data.  Run simulation first.")
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        # 获取数据
        if frame_idx < 0:
            frame_idx = len(self.history['field_snapshots']) + frame_idx

        snapshot_idx = min(frame_idx, len(self.history['field_snapshots']) - 1)
        data_idx = min(frame_idx * 2, len(self.history['positions']) - 1)

        field = self.history['field_snapshots'][snapshot_idx]
        positions = self.history['positions'][data_idx]
        current_time = self.history['time'][data_idx]
        hotspot_pos = self.history['hotspot_positions'][snapshot_idx]

        # 绘制敏感度场
        im = ax.imshow(field,
                       extent=[self.domain[0], self.domain[1],
                               self.domain[2], self.domain[3]],
                       origin='lower', cmap='YlOrRd', alpha=0.7)
        plt.colorbar(im, ax=ax, label='Sensitivity', shrink=0.8)

        # 绘制热点中心
        for hp in hotspot_pos:
            ax.scatter(hp[0], hp[1], c='red', s=150, marker='*',
                       edgecolors='darkred', linewidths=1, zorder=15, label='Hotspot')

        # 绘制Voronoi质心和连线
        if show_centroids:
            cells = self.voronoi.compute_voronoi(positions, field)
            for i, cell in enumerate(cells):
                ax.scatter(cell.centroid[0], cell.centroid[1],
                           c=[colors[i]], s=80, marker='x', linewidths=3, zorder=12)
                ax.plot([positions[i, 0], cell.centroid[0]],
                        [positions[i, 1], cell.centroid[1]],
                        color=colors[i], linestyle='--', linewidth=1.5, alpha = 0.7)

                # 绘制轨迹和无人机
                for i in range(self.num_agents):
                    # 历史轨迹
                    traj_end = data_idx + 1
                    traj_start = max(0, traj_end - 150)
                    trajectory = np.array([self.history['positions'][j][i]
                                           for j in range(traj_start, traj_end)])

                    if len(trajectory) > 1:
                        # 渐变轨迹
                        points = trajectory.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        alphas = np.linspace(0.1, 0.8, len(segments))

                        for seg, alpha in zip(segments, alphas):
                            ax.plot(seg[:, 0], seg[:, 1], color=colors[i],
                                    alpha=alpha, linewidth=2)

                    # 起点
                    start_pos = self.history['positions'][0][i]
                    ax.scatter(start_pos[0], start_pos[1], c=[colors[i]], s=100,
                               marker='s', alpha=0.5, edgecolors='black', linewidths=1, zorder=10)

                    # 当前位置
                    ax.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=200,
                               marker='o', edgecolors='black', linewidths=2, zorder=13,
                               label=f'UAV {i}')

                ax.set_xlim(self.domain[0], self.domain[1])
                ax.set_ylim(self.domain[2], self.domain[3])
                ax.set_xlabel('X (m)', fontsize=12)
                ax.set_ylabel('Y (m)', fontsize=12)
                ax.set_title(f'UAV Coverage Control - t = {current_time:.1f}s', fontsize=14, fontweight='bold')
                ax.set_aspect('equal')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"Saved:  {save_path}")

                return fig, ax

    def plot_keyframes(self, num_frames: int = 6, save_dir: str = None):
        """
    绘制关键帧序列

    Args:
        num_frames: 关键帧数量
        save_dir: 保存目录
    """
        if not self.history['field_snapshots']:
            print("No data. Run simulation first.")
            return

        total_snapshots = len(self.history['field_snapshots'])
        indices = np.linspace(0, total_snapshots - 1, num_frames, dtype=int)

        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "keyframes")
        os.makedirs(save_dir, exist_ok=True)

        print(f"Generating {num_frames} keyframes...")

        for i, idx in enumerate(indices):
            save_path = os.path.join(save_dir, f"frame_{i: 03d}.png")
            self.plot_trajectory_frame(frame_idx=idx, save_path=save_path)
            plt.close()

        print(f"Keyframes saved to: {save_dir}")

    def plot_coverage_curves(self, save_path: str = None):
        """
    绘制覆盖代价和覆盖率曲线

    Args:
        save_path: 保存路径
    """
        if not self.history['time']:
            print("No data.  Run simulation first.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        time = np.array(self.history['time'])
        cost = np.array(self.history['coverage_cost'])
        coverage = np.array(self.history['weighted_coverage'])

        # 图1: 覆盖代价
        ax1 = axes[0]
        ax1.plot(time, cost, 'b-', linewidth=2, label='Coverage Cost')
        ax1.fill_between(time, cost, alpha=0.2)

        # 标注最小值
        min_idx = np.argmin(cost)
        ax1.scatter(time[min_idx], cost[min_idx], c='green', s=100, zorder=10)
        ax1.annotate(f'Min: {cost[min_idx]:.0f}',
                     (time[min_idx], cost[min_idx]),
                     textcoords="offset points", xytext=(10, 10),
                     fontsize=10, color='green', fontweight='bold')

        ax1.axhline(y=cost[min_idx], color='g', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Coverage Cost', fontsize=12)
        ax1.set_title('Coverage Cost Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # 图2: 加权覆盖率
        ax2 = axes[1]
        ax2.plot(time, coverage * 100, 'r-', linewidth=2, label='Weighted Coverage')
        ax2.fill_between(time, coverage * 100, alpha=0.2, color='red')

        # 标注最大值
        max_idx = np.argmax(coverage)
        ax2.scatter(time[max_idx], coverage[max_idx] * 100, c='darkred', s=100, zorder=10)
        ax2.annotate(f'Max: {coverage[max_idx]:.1%}',
                     (time[max_idx], coverage[max_idx] * 100),
                     textcoords="offset points", xytext=(10, -15),
                     fontsize=10, color='darkred', fontweight='bold')

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Weighted Coverage (%)', fontsize=12)
        ax2.set_title('Weighted Coverage Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right')
        ax2.set_ylim(0, 100)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "coverage_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

        plt.show()
        return fig, axes

    def create_animation(self, save_path: str = None, fps: int = 15, dpi: int = 100):
        """
    创建动画 (GIF)

    Args:
        save_path: 保存路径
        fps: 帧率
        dpi:  分辨率
    """
        if not self.history['field_snapshots']:
            print("No data. Run simulation first.")
            return

        print("Creating animation...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        num_frames = len(self.history['field_snapshots'])
        cost_min = min(self.history['coverage_cost']) * 0.9
        cost_max = max(self.history['coverage_cost']) * 1.1

        def animate(frame_idx):
            for ax in axes:
                ax.clear()

            data_idx = min(frame_idx * 2, len(self.history['positions']) - 1)
            field = self.history['field_snapshots'][frame_idx]
            positions = self.history['positions'][data_idx]
            current_time = self.history['time'][data_idx]
            hotspot_pos = self.history['hotspot_positions'][frame_idx]

            # 左图: 轨迹
            ax1 = axes[0]
            ax1.imshow(field, extent=[self.domain[0], self.domain[1],
                                      self.domain[2], self.domain[3]],
                       origin='lower', cmap='YlOrRd', alpha=0.7)

            # 热点
            for hp in hotspot_pos:
                ax1.scatter(hp[0], hp[1], c='red', s=100, marker='*', zorder=15)

            # Voronoi
            cells = self.voronoi.compute_voronoi(positions, field)

            for i in range(self.num_agents):
                pos = positions[i]
                cell = cells[i]

                # 轨迹
                traj_end = data_idx + 1
                traj_start = max(0, traj_end - 100)
                trajectory = np.array([self.history['positions'][j][i]
                                       for j in range(traj_start, traj_end)])
                if len(trajectory) > 1:
                    ax1.plot(trajectory[:, 0], trajectory[:, 1],
                             color=colors[i], alpha=0.5, linewidth=1.5)

                # UAV
                ax1.scatter(pos[0], pos[1], c=[colors[i]], s=150, marker='o',
                            edgecolors='black', linewidths=2, zorder=13)

                # 质心连线
                ax1.scatter(cell.centroid[0], cell.centroid[1],
                            c=[colors[i]], s=60, marker='x', linewidths=2, zorder=12)
                ax1.plot([pos[0], cell.centroid[0]], [pos[1], cell.centroid[1]],
                         color=colors[i], linestyle='--', alpha=0.6)

            ax1.set_xlim(self.domain[0], self.domain[1])
            ax1.set_ylim(self.domain[2], self.domain[3])
            ax1.set_title(f'Coverage Control - t = {current_time:.1f}s', fontweight='bold')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_aspect('equal')

            # 右图: 代价曲线
            ax2 = axes[1]
            time_data = self.history['time'][: data_idx + 1]
            cost_data = self.history['coverage_cost'][:data_idx + 1]

            ax2.plot(time_data, cost_data, 'b-', linewidth=2)
            ax2.fill_between(time_data, cost_data, alpha=0.3)

            if cost_data:
                ax2.scatter(time_data[-1], cost_data[-1], c='red', s=100, zorder=10)

            ax2.set_xlim(0, self.total_time)
            ax2.set_ylim(cost_min, cost_max)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Coverage Cost')
            ax2.set_title('Coverage Cost', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

            plt.tight_layout()

            if frame_idx % 20 == 0:
                print(f"  Frame {frame_idx}/{num_frames}")

            return axes

        anim = FuncAnimation(fig, animate, frames=num_frames,
                             interval=1000 // fps, blit=False)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "coverage_animation.gif")

        print(f"Saving animation to {save_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        print(f"Animation saved:  {save_path}")

        plt.close(fig)
        return save_path

    def print_statistics(self):
        """打印仿真统计"""
        if not self.history['coverage_cost']:
            print("No data.")
            return

        costs = self.history['coverage_cost']
        coverage = self.history['weighted_coverage']

        print("\n" + "=" * 50)
        print("SIMULATION STATISTICS")
        print("=" * 50)
        print(f"Duration: {self.total_time}s, Steps: {len(costs)}")
        print(f"UAVs: {self.num_agents}, Field:  {self.field_preset}")
        print("-" * 50)
        print("Coverage Cost:")
        print(f"  Initial: {costs[0]:.0f}")
        print(f"  Final:    {costs[-1]:.0f}")
        print(f"  Min:     {min(costs):.0f}")
        print(f"  Max:     {max(costs):.0f}")
        print(f"  Avg:     {np.mean(costs):.0f}")
        improvement = (costs[0] - costs[-1]) / costs[0] * 100
        print(f"  Improvement: {improvement:.1f}%")
        print("-" * 50)
        print("Weighted Coverage:")
        print(f"  Initial: {coverage[0]:.1%}")
        print(f"  Final:    {coverage[-1]:.1%}")
        print(f"  Max:     {max(coverage):.1%}")
        print(f"  Avg:     {np.mean(coverage):.1%}")
        print("=" * 50)



def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-UAV Coverage Simulation')
    parser.add_argument('--agents', type=int, default=5, help='Number of UAVs')
    parser.add_argument('--time', type=float, default=30.0, help='Simulation time (s)')
    parser.add_argument('--field', type=str, default='mixed',
                        choices=['static', 'linear', 'circular', 'mixed'],
                        help='Field preset')
    parser.add_argument('--no-cbf', action='store_true', help='Disable CBF')
    parser.add_argument('--no-gp', action='store_true', help='Disable GP prediction')
    parser.add_argument('--keyframes', type=int, default=6, help='Number of keyframes')
    parser.add_argument('--no-animation', action='store_true', help='Skip animation generation')
    parser.add_argument('--fps', type=int, default=15, help='Animation FPS')

    args = parser.parse_args()

    # 创建仿真器
    sim = CoverageSimulator(
        num_agents=args.agents,
        total_time=args.time,
        field_preset=args.field,
        use_cbf=not args.no_cbf,
        use_gp_prediction=not args.no_gp
    )

    print("=" * 50)
    print("Multi-UAV Coverage Control Simulation")
    print("=" * 50)
    print(f"UAVs: {args.agents}")
    print(f"Time: {args.time}s")
    print(f"Field: {args.field}")
    print(f"CBF:  {'Enabled' if not args.no_cbf else 'Disabled'}")
    print(f"GP:   {'Enabled' if not args.no_gp else 'Disabled'}")
    print("=" * 50)

    # 运行仿真
    sim.run(verbose=True)

    # 打印统计
    sim.print_statistics()

    # 生成输出
    print("\nGenerating outputs...")

    # 1. 关键帧
    sim.plot_keyframes(num_frames=args.keyframes)

    # 2. 覆盖曲线
    sim.plot_coverage_curves()

    # 3. 动画（可选）
    if not args.no_animation:
        sim.create_animation(fps=args.fps)

    print("\nDone!")

if __name__ == "__main__":
    main()