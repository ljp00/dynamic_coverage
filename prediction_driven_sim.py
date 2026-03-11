"""
对比实验：预测驱动 vs 无预测（仅局部观测）
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage.prediction_driven_controller import (
    PredictionDrivenController,
    ReactiveLocalController
)
from src.coverage.voronoi import WeightedVoronoi
from src.safety.cbf import DistributedCBF
from src.agents.uav import UAVSwarm
from src.prediction.gp_predictor import SpatioTemporalGP


class DynamicHotspotField:
    """动态热点场"""

    def __init__(self, domain=(0, 100, 0, 100), resolution=50,
                 motion_type="circular", seed=None):
        self.domain = domain
        self.resolution = resolution
        self.motion_type = motion_type
        self.time = 0.0

        if seed is not None:
            np.random.seed(seed)

        self.x_grid = np.linspace(domain[0], domain[1], resolution)
        self.y_grid = np.linspace(domain[2], domain[3], resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

        self.intensity = 2.0
        self.spread = 15.0

        cx, cy = 50, 50
        self.origin = np.array([cx, cy])
        self.amplitude = np.array([25.0, 25.0])
        self.velocity = 0.2
        self.center = self.origin.copy()

    def update(self, dt):
        self.time += dt
        if self.motion_type == "circular":
            angle = self.velocity * self.time
            self.center = self.origin + np.array([
                self.amplitude[0] * np.cos(angle),
                self.amplitude[1] * np.sin(angle)
            ])
        elif self.motion_type == "linear":
            self.center = self.origin + np.array([
                self.amplitude[0] * np.sin(self.velocity * self.time), 0
            ])

    def get_position_at_time(self, t):
        if self.motion_type == "circular":
            angle = self.velocity * t
            return self.origin + np.array([
                self.amplitude[0] * np.cos(angle),
                self.amplitude[1] * np.sin(angle)
            ])
        return self.center.copy()

    def get_field_at_time(self, t):
        center = self.get_position_at_time(t)
        positions = np.column_stack([self.X.ravel(), self.Y.ravel()])
        dist_sq = np.sum((positions - center) ** 2, axis=1)
        density = self.intensity * np.exp(-dist_sq / (2 * self.spread ** 2))
        density += 0.1
        return density.reshape(self.resolution, self.resolution)

    def get_density(self, positions):
        positions = np.atleast_2d(positions)
        dist_sq = np.sum((positions - self.center) ** 2, axis=1)
        density = self.intensity * np.exp(-dist_sq / (2 * self.spread ** 2))
        density += 0.1
        return density

    def get_field_grid(self):
        return self.get_field_at_time(self.time)

    def get_hotspot_position(self):
        return self.center.copy()


class PredictiveSimulator:
    """有预测的仿真器"""

    def __init__(self, field, num_agents=3):
        self.field = field
        self.domain = field.domain
        self.num_agents = num_agents

        self.swarm = UAVSwarm(num_agents, self.domain, max_velocity=5.0, sensing_radius=20.0)
        self.controller = PredictionDrivenController(
            domain=self.domain,
            max_velocity=5.0,
            attraction_gain=1.5,
            repulsion_gain=2.0,
            repulsion_radius=12.0
        )
        self.cbf = DistributedCBF(safe_distance=3.0, gamma=1.0,
                                  domain=self.domain, communication_radius=30.0)

        # GP预测器
        self.gp = SpatioTemporalGP(length_scale_space=15.0, length_scale_time=8.0, noise_variance=0.1)
        self.gp_trained = False
        self.prediction_horizon = 10

        self.time = 0.0
        self.dt = 0.1
        self.step_count = 0

        self.history = {
            'time': [],
            'positions': [],
            'hotspot_distance': [],
            'coverage_cost': [],
            'predicted_field': [],
            'prediction_error': [],
        }

    def step(self, dt):
        self.dt = dt
        current_field = self.field.get_field_grid()
        positions = self.swarm.get_positions()
        current_hotspot = self.field.get_hotspot_position()

        # GP学习
        self.swarm.sense_all(self.field, self.time)
        X, y = self.swarm.get_all_sensed_data(time_window=25.0, current_time=self.time)

        if self.step_count % 5 == 0 and len(X) >= 20:
            try:
                self.gp.fit(X, y)
                self.gp_trained = True
            except:
                pass

        # 预测
        prediction_error = 0.0
        if self.gp_trained:
            future_time = self.time + self.prediction_horizon * dt
            predicted_field = self._predict_field(future_time)

            # 计算预测误差
            predicted_hotspot = self._find_hotspot(predicted_field)
            actual_future_hotspot = self.field.get_position_at_time(future_time)
            prediction_error = np.linalg.norm(predicted_hotspot - actual_future_hotspot)

            prediction_weight = 0.7
        else:
            predicted_field = current_field
            prediction_weight = 0.0

        # 控制
        controls = self.controller.compute_control(
            positions, current_field, predicted_field, prediction_weight
        )
        controls = self.cbf.filter_control(positions, controls)

        self.swarm.update_all(controls, dt, use_velocity=True)

        # 记录
        if self.step_count % 2 == 0:
            self._record(current_field, predicted_field, current_hotspot, prediction_error)

        self.time += dt
        self.step_count += 1

    def _predict_field(self, future_time):
        x_grid = np.linspace(self.domain[0], self.domain[1], 50)
        y_grid = np.linspace(self.domain[2], self.domain[3], 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        X_pred = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, future_time)])
        try:
            mean, _ = self.gp.predict(X_pred)
            return np.maximum(mean.reshape(50, 50), 0.1)
        except:
            return self.field.get_field_grid()

    def _find_hotspot(self, field):
        max_idx = np.argmax(field)
        row, col = np.unravel_index(max_idx, field.shape)
        x = self.domain[0] + (self.domain[1] - self.domain[0]) * col / (field.shape[1] - 1)
        y = self.domain[2] + (self.domain[3] - self.domain[2]) * row / (field.shape[0] - 1)
        return np.array([x, y])

    def _record(self, current_field, predicted_field, hotspot, prediction_error):
        positions = self.swarm.get_positions()
        distances = np.linalg.norm(positions - hotspot, axis=1)

        voronoi = WeightedVoronoi(self.domain, resolution=50)
        cost = voronoi.compute_coverage_cost(positions, current_field)

        self.history['time'].append(self.time)
        self.history['positions'].append(positions.copy())
        self.history['hotspot_distance'].append(np.min(distances))
        self.history['coverage_cost'].append(cost)
        self.history['predicted_field'].append(predicted_field.copy())
        self.history['prediction_error'].append(prediction_error)


class ReactiveSimulator:
    """无预测（仅局部观测）的仿真器"""

    def __init__(self, field, num_agents=3):
        self.field = field
        self.domain = field.domain
        self.num_agents = num_agents

        self.swarm = UAVSwarm(num_agents, self.domain, max_velocity=5.0, sensing_radius=20.0)
        self.controller = ReactiveLocalController(
            domain=self.domain,
            max_velocity=5.0,
            sensing_radius=20.0,
            gradient_gain=2.0,
            repulsion_gain=2.0
        )
        self.cbf = DistributedCBF(safe_distance=3.0, gamma=1.0,
                                  domain=self.domain, communication_radius=30.0)

        self.time = 0.0
        self.step_count = 0

        self.history = {
            'time': [],
            'positions': [],
            'hotspot_distance': [],
            'coverage_cost': [],
        }

    def step(self, dt):
        current_field = self.field.get_field_grid()
        positions = self.swarm.get_positions()
        hotspot = self.field.get_hotspot_position()

        # 反应式控制（仅基于当前局部观测）
        controls = self.controller.compute_control(positions, current_field)
        controls = self.cbf.filter_control(positions, controls)

        self.swarm.update_all(controls, dt, use_velocity=True)

        # 记录
        if self.step_count % 2 == 0:
            self._record(current_field, hotspot)

        self.time += dt
        self.step_count += 1

    def _record(self, current_field, hotspot):
        positions = self.swarm.get_positions()
        distances = np.linalg.norm(positions - hotspot, axis=1)

        voronoi = WeightedVoronoi(self.domain, resolution=50)
        cost = voronoi.compute_coverage_cost(positions, current_field)

        self.history['time'].append(self.time)
        self.history['positions'].append(positions.copy())
        self.history['hotspot_distance'].append(np.min(distances))
        self.history['coverage_cost'].append(cost)


class PredictionVsReactiveExperiment:
    """预测 vs 反应式 对比实验"""

    def __init__(self, num_agents=3, total_time=80.0, dt=0.1,
                 motion_type="circular", seed=42):
        self.num_agents = num_agents
        self.total_time = total_time
        self.dt = dt
        self.motion_type = motion_type
        self.seed = seed
        self.domain = (0, 100, 0, 100)

        self.field = DynamicHotspotField(
            domain=self.domain, resolution=50, motion_type=motion_type, seed=seed
        )

        # 有预测
        np.random.seed(seed)
        self.sim_predictive = PredictiveSimulator(self.field, num_agents)

        # 无预测（仅局部观测）
        np.random.seed(seed)
        self.sim_reactive = ReactiveSimulator(self.field, num_agents)

        self.time = 0.0
        self.field_history = []
        self.hotspot_history = []

        self.output_dir = "output/prediction_vs_reactive"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, verbose=True):
        num_steps = int(self.total_time / self.dt)

        if verbose:
            print("=" * 70)
            print("Prediction-Driven vs Reactive (Local Observation Only)")
            print("=" * 70)
            print("Motion type: %s" % self.motion_type)
            print("UAVs: %d, Time: %.0fs" % (self.num_agents, self.total_time))
            print("-" * 70)

        for i in range(num_steps):
            self.field.update(self.dt)
            self.sim_predictive.step(self.dt)
            self.sim_reactive.step(self.dt)

            if i % 2 == 0:
                self.field_history.append(self.field.get_field_grid().copy())
                self.hotspot_history.append(self.field.get_hotspot_position().copy())

            self.time += self.dt

            if verbose and i % 100 == 0:
                dist_pred = self.sim_predictive.history['hotspot_distance'][-1] if self.sim_predictive.history['hotspot_distance'] else 0
                dist_react = self.sim_reactive.history['hotspot_distance'][-1] if self.sim_reactive.history['hotspot_distance'] else 0
                print("Step %4d: Predictive=%.1fm, Reactive=%.1fm" % (i, dist_pred, dist_react))

        if verbose:
            print("-" * 70)
            print("Complete!")

    def print_statistics(self):
        h_pred = self.sim_predictive.history
        h_react = self.sim_reactive.history

        # 后半段统计
        half = len(h_pred['time']) // 2

        print("\n" + "=" * 70)
        print("COMPARISON RESULTS (Second Half Statistics)")
        print("=" * 70)
        print("%-30s %15s %15s" % ("Metric", "Predictive", "Reactive"))
        print("-" * 70)

        # 热点距离
        dist_pred = np.mean(h_pred['hotspot_distance'][half:])
        dist_react = np.mean(h_react['hotspot_distance'][half:])
        print("%-30s %15.1f %15.1f" % ("Hotspot Distance (Avg)", dist_pred, dist_react))

        dist_pred_min = np.min(h_pred['hotspot_distance'][half:])
        dist_react_min = np.min(h_react['hotspot_distance'][half:])
        print("%-30s %15.1f %15.1f" % ("Hotspot Distance (Min)", dist_pred_min, dist_react_min))

        # 覆盖代价
        cost_pred = np.mean(h_pred['coverage_cost'][half:])
        cost_react = np.mean(h_react['coverage_cost'][half:])
        print("%-30s %15.0f %15.0f" % ("Coverage Cost (Avg)", cost_pred, cost_react))

        # 预测误差
        pred_errors = np.array(h_pred['prediction_error'][half:])
        valid_errors = pred_errors[pred_errors > 0]
        if len(valid_errors) > 0:
            print("%-30s %15.1f %15s" % ("Prediction Error (Avg)", np.mean(valid_errors), "N/A"))

        print("-" * 70)

        # 改善率
        improvement_dist = (dist_react - dist_pred) / dist_react * 100
        improvement_cost = (cost_react - cost_pred) / cost_react * 100

        print("%-30s %14.1f%%" % ("Improvement (Distance)", improvement_dist))
        print("%-30s %14.1f%%" % ("Improvement (Cost)", improvement_cost))

        print("=" * 70)

        # 结论
        print("\nCONCLUSION:")
        if improvement_dist > 5:
            print("  ✓ Prediction-driven control shows SIGNIFICANT improvement!")
            print("  ✓ Predictive approach tracks hotspot %.1f%% better" % improvement_dist)
        elif improvement_dist > 0:
            print("  ○ Prediction-driven control shows SLIGHT improvement")
        else:
            print("  ✗ Reactive control performs better in this scenario")
            print("    Consider: longer simulation, different motion pattern")

    def plot_comparison(self, save_path=None, show=True):
        """绘制对比曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        h_pred = self.sim_predictive.history
        h_react = self.sim_reactive.history

        time_pred = np.array(h_pred['time'])
        time_react = np.array(h_react['time'])

        # 图1: 热点距离对比
        ax1 = axes[0, 0]
        ax1.plot(time_pred, h_pred['hotspot_distance'], 'b-', linewidth=2, label='Predictive')
        ax1.plot(time_react, h_react['hotspot_distance'], 'r--', linewidth=2, label='Reactive (Local)')
        ax1.fill_between(time_pred, h_pred['hotspot_distance'], alpha=0.2, color='blue')
        ax1.fill_between(time_react, h_react['hotspot_distance'], alpha=0.2, color='red')
        ax1.axvline(x=time_pred[len(time_pred)//2], color='gray', linestyle=':', alpha=0.7, label='Stats start')
        ax1.set_ylabel('Min Distance to Hotspot (m)')
        ax1.set_title('Hotspot Tracking Performance', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 图2: 覆盖代价对比
        ax2 = axes[0, 1]
        ax2.plot(time_pred, h_pred['coverage_cost'], 'b-', linewidth=2, label='Predictive')
        ax2.plot(time_react, h_react['coverage_cost'], 'r--', linewidth=2, label='Reactive (Local)')
        ax2.axvline(x=time_pred[len(time_pred)//2], color='gray', linestyle=':', alpha=0.7)
        ax2.set_ylabel('Coverage Cost')
        ax2.set_title('Coverage Cost Over Time', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # 图3: 预测误差
        ax3 = axes[1, 0]
        pred_errors = np.array(h_pred['prediction_error'])
        ax3.plot(time_pred, pred_errors, 'purple', linewidth=2, label='Prediction Error')
        ax3.fill_between(time_pred, pred_errors, alpha=0.3, color='purple')
        ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Threshold (10m)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Prediction Error (m)')
        ax3.set_title('GP Prediction Error', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 图4: 性能对比柱状图
        ax4 = axes[1, 1]
        half = len(h_pred['time']) // 2

        metrics = ['Avg Distance', 'Min Distance']
        pred_vals = [np.mean(h_pred['hotspot_distance'][half:]),
                    np.min(h_pred['hotspot_distance'][half:])]
        react_vals = [np.mean(h_react['hotspot_distance'][half:]),
                     np.min(h_react['hotspot_distance'][half:])]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax4.bar(x - width/2, pred_vals, width, label='Predictive', color='blue', alpha=0.7)
        bars2 = ax4.bar(x + width/2, react_vals, width, label='Reactive', color='red', alpha=0.7)

        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.set_ylabel('Distance (m)')
        ax4.set_title('Performance Summary (2nd Half)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 标注数值
        for bar, val in zip(bars1, pred_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    '%.1f' % val, ha='center', va='bottom', fontsize=10, color='blue')
        for bar, val in zip(bars2, react_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    '%.1f' % val, ha='center', va='bottom', fontsize=10, color='red')

        plt.suptitle('Prediction-Driven vs Reactive Control (%s motion)' % self.motion_type,
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved: %s" % save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_trajectory_comparison(self, frame_idx=-1, save_path=None, show=True):
        """绘制轨迹对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if frame_idx < 0:
            frame_idx = len(self.field_history) + frame_idx
        frame_idx = min(frame_idx, len(self.field_history) - 1)

        data_idx = min(frame_idx * 2, len(self.sim_predictive.history['time']) - 1)

        field = self.field_history[frame_idx]
        hotspot = self.hotspot_history[frame_idx]
        current_time = self.sim_predictive.history['time'][data_idx]

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        sims = [
            (self.sim_predictive, "Predictive (GP Prediction)"),
            (self.sim_reactive, "Reactive (Local Observation)")
        ]

        for ax_idx, (sim, title) in enumerate(sims):
            ax = axes[ax_idx]

            # 绘制场
            ax.imshow(field, extent=[0, 100, 0, 100], origin='lower',
                     cmap='YlOrRd', alpha=0.7, vmin=0, vmax=2.5)

            # 热点
            ax.scatter(hotspot[0], hotspot[1], c='red', s=200, marker='*',
                      edgecolors='darkred', linewidths=2, zorder=20)
            circle = plt.Circle(hotspot, 15, fill=False, color='red',
                               linestyle='--', linewidth=2, alpha=0.5)
            ax.add_patch(circle)

            # 轨迹和无人机
            positions = sim.history['positions'][data_idx]

            for i in range(self.num_agents):
                # 轨迹
                traj_end = data_idx + 1
                traj_start = max(0, traj_end - 60)
                trajectory = np.array([sim.history['positions'][j][i]
                                      for j in range(traj_start, traj_end)])

                if len(trajectory) > 1:
                    alphas = np.linspace(0.1, 0.8, len(trajectory) - 1)
                    for k in range(len(trajectory) - 1):
                        ax.plot(trajectory[k:k+2, 0], trajectory[k:k+2, 1],
                               color=colors[i], alpha=alphas[k], linewidth=2)

                # 起点
                start = sim.history['positions'][0][i]
                ax.scatter(start[0], start[1], c=[colors[i]], s=80,
                          marker='s', alpha=0.5, edgecolors='black', zorder=10)

                # 当前位置
                ax.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=150,
                          marker='o', edgecolors='black', linewidths=2, zorder=15)

            dist = sim.history['hotspot_distance'][data_idx]
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('%s\nt=%.1fs, Dist=%.1fm' % (title, current_time, dist),
                        fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "trajectory_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved: %s" % save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def create_animation(self, save_path=None, fps=10):
        """创建动画"""
        print("Creating animation...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
        num_frames = len(self.field_history)

        print("Total frames: %d" % num_frames)

        def animate(frame_idx):
            for ax in axes:
                ax.clear()

            data_idx = min(frame_idx * 2, len(self.sim_predictive.history['time']) - 1)
            field = self.field_history[frame_idx]
            hotspot = self.hotspot_history[frame_idx]
            current_time = self.sim_predictive.history['time'][data_idx]

            sims = [
                (self.sim_predictive, "Predictive"),
                (self.sim_reactive, "Reactive")
            ]

            for ax_idx, (sim, title) in enumerate(sims):
                ax = axes[ax_idx]

                ax.imshow(field, extent=[0, 100, 0, 100], origin='lower',
                         cmap='YlOrRd', alpha=0.7, vmin=0, vmax=2.5)

                ax.scatter(hotspot[0], hotspot[1], c='red', s=200, marker='*',
                          edgecolors='darkred', linewidths=2, zorder=20)
                circle = plt.Circle(hotspot, 15, fill=False, color='red',
                                   linestyle='--', linewidth=2, alpha=0.5)
                ax.add_patch(circle)

                positions = sim.history['positions'][data_idx]

                for i in range(self.num_agents):
                    traj_end = data_idx + 1
                    traj_start = max(0, traj_end - 40)
                    trajectory = np.array([sim.history['positions'][j][i]
                                          for j in range(traj_start, traj_end)])
                    if len(trajectory) > 1:
                        ax.plot(trajectory[:, 0], trajectory[:, 1],
                               color=colors[i], alpha=0.5, linewidth=1.5)

                    ax.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=120,
                              marker='o', edgecolors='black', linewidths=2, zorder=15)

                dist = sim.history['hotspot_distance'][data_idx]
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.set_title('%s | Dist=%.1fm' % (title, dist), fontsize=11, fontweight='bold')
                ax.set_aspect('equal')

            plt.suptitle('t = %.1fs' % current_time, fontsize=12, fontweight='bold')
            plt.tight_layout()

            if frame_idx % 30 == 0:
                print("  Frame %d/%d" % (frame_idx, num_frames))

            return axes

        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000//fps, blit=False)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "comparison_animation.gif")

        print("Saving to %s..." % save_path)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        print("Animation saved: %s" % save_path)

        plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prediction vs Reactive Control')
    parser.add_argument('--agents', type=int, default=3)
    parser.add_argument('--time', type=float, default=80.0)
    parser.add_argument('--motion', type=str, default='circular',
                       choices=['circular', 'linear'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-animation', action='store_true')
    parser.add_argument('--fps', type=int, default=10)

    args = parser.parse_args()

    print("=" * 70)
    print("PREDICTION-DRIVEN vs REACTIVE CONTROL")
    print("=" * 70)
    print("")
    print("Predictive: Uses GP to predict future field, moves proactively")
    print("Reactive:   Only uses local observation, moves reactively")
    print("")
    print("Settings:")
    print("  UAVs: %d" % args.agents)
    print("  Time: %.0fs" % args.time)
    print("  Motion: %s" % args.motion)
    print("=" * 70)

    exp = PredictionVsReactiveExperiment(
        num_agents=args.agents,
        total_time=args.time,
        motion_type=args.motion,
        seed=args.seed
    )

    exp.run(verbose=True)
    exp.print_statistics()

    print("\nGenerating visualizations...")
    exp.plot_comparison(show=False)
    exp.plot_trajectory_comparison(show=False)

    if not args.no_animation:
        exp.create_animation(fps=args.fps)

    print("\nOutputs saved to: %s" % exp.output_dir)

    # 显示结果
    exp.plot_comparison(show=True)


if __name__ == "__main__":
    main()