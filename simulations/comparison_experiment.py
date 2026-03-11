"""
GP预测可视化对比实验
显示：真实场 vs GP预测场 vs 无人机轨迹
"""
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage.voronoi import WeightedVoronoi
from src.coverage.lloyd_controller import LloydController
from src.safety.cbf import DistributedCBF
from src.agents.uav import UAVSwarm
from src.prediction.gp_predictor import SpatioTemporalGP


class MovingHotspot:
    """移动热点场"""

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

        cx = (domain[0] + domain[1]) / 2
        cy = (domain[2] + domain[3]) / 2
        self.origin = np.array([cx, cy])

        if motion_type == "circular":
            self.amplitude = np.array([25.0, 25.0])
            self.velocity = 0.2
        elif motion_type == "linear":
            self.amplitude = np.array([30.0, 0.0])
            self.velocity = 0.3
        else:
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
            self.center = self.origin + self.amplitude * np.sin(self.velocity * self.time)

    def get_position_at_time(self, t):
        """获取任意时刻的热点位置"""
        if self.motion_type == "circular":
            angle = self.velocity * t
            return self.origin + np.array([
                self.amplitude[0] * np.cos(angle),
                self.amplitude[1] * np.sin(angle)
            ])
        elif self.motion_type == "linear":
            return self.origin + self.amplitude * np.sin(self.velocity * t)
        return self.center.copy()

    def get_field_at_time(self, t):
        """获取任意时刻的敏感度场"""
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


class PredictionVisualizer:
    """带预测场可视化的仿真器"""

    def __init__(self, field, num_agents=3, use_gp=True):
        self.field = field
        self.domain = field.domain
        self.num_agents = num_agents
        self.use_gp = use_gp

        self.swarm = UAVSwarm(num_agents, self.domain, max_velocity=5.0, sensing_radius=15.0)
        self.controller = LloydController(self.domain, gain=1.5, max_velocity=5.0, resolution=50)
        self.voronoi = WeightedVoronoi(self.domain, resolution=50)
        self.cbf = DistributedCBF(safe_distance=3.0, gamma=1.0, domain=self.domain, communication_radius=30.0)

        if use_gp:
            self.gp = SpatioTemporalGP(length_scale_space=15.0, length_scale_time=10.0, noise_variance=0.1)
            self.gp_trained = False
            self.prediction_horizon = 10

        self.time = 0.0
        self.dt = 0.1
        self.step_count = 0

        self.history = {
            'time': [],
            'positions': [],
            'coverage_cost': [],
            'hotspot_distance': [],
            # 场快照
            'current_field': [],  # 当前真实场
            'predicted_field': [],  # GP预测场
            'future_actual_field': [],  # 未来真实场（用于对比）
            'hotspot_pos': [],  # 当前热点位置
            'predicted_hotspot_pos': [],  # 预测的热点位置
            'future_hotspot_pos': [],  # 未来真实热点位置
            'prediction_error': [],  # 预测误差
        }

    def step(self, dt):
        self.dt = dt
        current_field = self.field.get_field_grid()
        positions = self.swarm.get_positions()
        current_hotspot = self.field.get_hotspot_position()

        # 预测相关
        predicted_field = None
        predicted_hotspot = None
        future_actual_field = None
        future_hotspot = None
        prediction_error = 0.0

        if self.use_gp:
            self.swarm.sense_all(self.field, self.time)

            X, y = self.swarm.get_all_sensed_data(time_window=20.0, current_time=self.time)

            if self.step_count % 5 == 0 and len(X) >= 15:
                try:
                    self.gp.fit(X, y)
                    self.gp_trained = True
                except:
                    pass

            if self.gp_trained:
                # 预测未来场
                future_time = self.time + self.prediction_horizon * dt
                predicted_field = self._predict_field_at_time(future_time)
                predicted_hotspot = self._find_hotspot_in_field(predicted_field)

                # 获取未来真实场（用于对比）
                future_actual_field = self.field.get_field_at_time(future_time)
                future_hotspot = self.field.get_position_at_time(future_time)

                # 计算预测误差
                prediction_error = np.linalg.norm(predicted_hotspot - future_hotspot)

                # 使用预测场计算控制
                control_field = predicted_field
                alpha = 0.6
            else:
                control_field = current_field
                alpha = 1.0
        else:
            control_field = current_field
            alpha = 1.0

        controls = self.controller.compute_predictive_control(
            positions, current_field, control_field, alpha=alpha
        )

        controls = self.cbf.filter_control(positions, controls)
        self.swarm.update_all(controls, dt, use_velocity=True)

        # 记录数据（每2步记录一次以节省内存）
        if self.step_count % 2 == 0:
            self._record(current_field, predicted_field, future_actual_field,
                         current_hotspot, predicted_hotspot, future_hotspot, prediction_error)

        self.time += dt
        self.step_count += 1

    def _predict_field_at_time(self, future_time):
        """预测指定时刻的敏感度场"""
        x_grid = np.linspace(self.domain[0], self.domain[1], 50)
        y_grid = np.linspace(self.domain[2], self.domain[3], 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        X_pred = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, future_time)])
        try:
            mean, _ = self.gp.predict(X_pred)
            return np.maximum(mean.reshape(50, 50), 0.1)
        except:
            return self.field.get_field_grid()

    def _find_hotspot_in_field(self, field):
        """从场中找到热点位置（最大值位置）"""
        max_idx = np.argmax(field)
        row, col = np.unravel_index(max_idx, field.shape)
        x = self.domain[0] + (self.domain[1] - self.domain[0]) * col / (field.shape[1] - 1)
        y = self.domain[2] + (self.domain[3] - self.domain[2]) * row / (field.shape[0] - 1)
        return np.array([x, y])

    def _record(self, current_field, predicted_field, future_actual_field,
                current_hotspot, predicted_hotspot, future_hotspot, prediction_error):
        positions = self.swarm.get_positions()

        cost = self.voronoi.compute_coverage_cost(positions, current_field)
        distances = np.linalg.norm(positions - current_hotspot, axis=1)
        min_dist = np.min(distances)

        self.history['time'].append(self.time)
        self.history['positions'].append(positions.copy())
        self.history['coverage_cost'].append(cost)
        self.history['hotspot_distance'].append(min_dist)

        # 场快照
        self.history['current_field'].append(current_field.copy())
        self.history['predicted_field'].append(predicted_field.copy() if predicted_field is not None else None)
        self.history['future_actual_field'].append(
            future_actual_field.copy() if future_actual_field is not None else None)
        self.history['hotspot_pos'].append(current_hotspot.copy())
        self.history['predicted_hotspot_pos'].append(
            predicted_hotspot.copy() if predicted_hotspot is not None else None)
        self.history['future_hotspot_pos'].append(future_hotspot.copy() if future_hotspot is not None else None)
        self.history['prediction_error'].append(prediction_error)


class PredictionComparisonExperiment:
    """预测场可视化对比实验"""

    def __init__(self, num_agents=3, total_time=60.0, dt=0.1, motion_type="circular", seed=42):
        self.num_agents = num_agents
        self.total_time = total_time
        self.dt = dt
        self.motion_type = motion_type
        self.seed = seed
        self.domain = (0, 100, 0, 100)

        self.field = MovingHotspot(
            domain=self.domain, resolution=50, motion_type=motion_type, seed=seed
        )

        np.random.seed(seed)
        self.sim_gp = PredictionVisualizer(self.field, num_agents, use_gp=True)

        np.random.seed(seed)
        self.sim_no_gp = PredictionVisualizer(self.field, num_agents, use_gp=False)

        self.time = 0.0

        self.output_dir = "output/prediction_viz"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, verbose=True):
        num_steps = int(self.total_time / self.dt)

        if verbose:
            print("Running prediction visualization experiment...")
            print("Total steps: %d" % num_steps)
            print("-" * 60)

        for i in range(num_steps):
            self.field.update(self.dt)
            self.sim_gp.step(self.dt)
            self.sim_no_gp.step(self.dt)

            self.time += self.dt

            if verbose and i % 100 == 0:
                h = self.sim_gp.history
                if h['prediction_error']:
                    pred_err = h['prediction_error'][-1]
                    dist_gp = h['hotspot_distance'][-1]
                    dist_no_gp = self.sim_no_gp.history['hotspot_distance'][-1]
                    print("Step %4d: PredErr=%.1f, Dist(GP)=%.1f, Dist(NoGP)=%.1f" %
                          (i, pred_err, dist_gp, dist_no_gp))

        if verbose:
            print("-" * 60)
            print("Complete!")

    def plot_field_comparison(self, frame_idx=-1, save_path=None, show=True):
        """
        绘制场对比图：当前场 vs GP预测场 vs 未来真实场
        """
        h = self.sim_gp.history

        if frame_idx < 0:
            frame_idx = len(h['time']) + frame_idx
        frame_idx = min(frame_idx, len(h['time']) - 1)

        current_time = h['time'][frame_idx]
        current_field = h['current_field'][frame_idx]
        predicted_field = h['predicted_field'][frame_idx]
        future_actual_field = h['future_actual_field'][frame_idx]

        current_hotspot = h['hotspot_pos'][frame_idx]
        predicted_hotspot = h['predicted_hotspot_pos'][frame_idx]
        future_hotspot = h['future_hotspot_pos'][frame_idx]

        positions_gp = h['positions'][frame_idx]
        positions_no_gp = self.sim_no_gp.history['positions'][frame_idx]

        # 创建图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        # 获取全局颜色范围
        vmin, vmax = 0, 2.5

        # ===== 第一行：场对比 =====

        # 图1: 当前真实场
        ax1 = axes[0, 0]
        im1 = ax1.imshow(current_field, extent=[0, 100, 0, 100], origin='lower',
                         cmap='YlOrRd', alpha=0.8, vmin=vmin, vmax=vmax)
        ax1.scatter(current_hotspot[0], current_hotspot[1], c='red', s=200, marker='*',
                    edgecolors='darkred', linewidths=2, zorder=20, label='Current Hotspot')
        ax1.set_title('Current Field (t=%.1fs)' % current_time, fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Sensitivity')

        # 图2: GP预测场
        ax2 = axes[0, 1]
        if predicted_field is not None:
            im2 = ax2.imshow(predicted_field, extent=[0, 100, 0, 100], origin='lower',
                             cmap='YlOrRd', alpha=0.8, vmin=vmin, vmax=vmax)
            if predicted_hotspot is not None:
                ax2.scatter(predicted_hotspot[0], predicted_hotspot[1], c='blue', s=200, marker='*',
                            edgecolors='darkblue', linewidths=2, zorder=20, label='Predicted Hotspot')
            ax2.set_title('GP Predicted Field (t+%.1fs)' % (self.sim_gp.prediction_horizon * self.dt),
                          fontsize=12, fontweight='bold')
            plt.colorbar(im2, ax=ax2, shrink=0.8, label='Sensitivity')
        else:
            ax2.text(0.5, 0.5, 'GP Not Yet Trained', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=14)
            ax2.set_title('GP Predicted Field (Not Available)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.set_aspect('equal')

        # 图3: 未来真实场
        ax3 = axes[0, 2]
        if future_actual_field is not None:
            im3 = ax3.imshow(future_actual_field, extent=[0, 100, 0, 100], origin='lower',
                             cmap='YlOrRd', alpha=0.8, vmin=vmin, vmax=vmax)
            if future_hotspot is not None:
                ax3.scatter(future_hotspot[0], future_hotspot[1], c='green', s=200, marker='*',
                            edgecolors='darkgreen', linewidths=2, zorder=20, label='Actual Future Hotspot')
            ax3.set_title('Actual Future Field (t+%.1fs)' % (self.sim_gp.prediction_horizon * self.dt),
                          fontsize=12, fontweight='bold')
            plt.colorbar(im3, ax=ax3, shrink=0.8, label='Sensitivity')
        else:
            ax3.text(0.5, 0.5, 'Not Available', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Actual Future Field', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_aspect('equal')

        # ===== 第二行：无人机轨迹对比 =====

        # 图4: GP控制的无人机
        ax4 = axes[1, 0]
        ax4.imshow(current_field, extent=[0, 100, 0, 100], origin='lower',
                   cmap='YlOrRd', alpha=0.5, vmin=vmin, vmax=vmax)
        ax4.scatter(current_hotspot[0], current_hotspot[1], c='red', s=150, marker='*',
                    edgecolors='darkred', linewidths=2, zorder=20)

        # 绘制轨迹
        for i in range(self.num_agents):
            traj_end = frame_idx + 1
            traj_start = max(0, traj_end - 50)
            trajectory = np.array([h['positions'][j][i] for j in range(traj_start, traj_end)])
            if len(trajectory) > 1:
                ax4.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], alpha=0.6, linewidth=2)
            ax4.scatter(positions_gp[i, 0], positions_gp[i, 1], c=[colors[i]], s=150,
                        marker='o', edgecolors='black', linewidths=2, zorder=15)

        ax4.set_title('With GP Prediction', fontsize=12, fontweight='bold')
        ax4.set_xlabel('X (m)')
        ax4.set_ylabel('Y (m)')
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
        ax4.set_aspect('equal')

        # 图5: 无GP控制的无人机
        ax5 = axes[1, 1]
        ax5.imshow(current_field, extent=[0, 100, 0, 100], origin='lower',
                   cmap='YlOrRd', alpha=0.5, vmin=vmin, vmax=vmax)
        ax5.scatter(current_hotspot[0], current_hotspot[1], c='red', s=150, marker='*',
                    edgecolors='darkred', linewidths=2, zorder=20)

        h_no_gp = self.sim_no_gp.history
        for i in range(self.num_agents):
            traj_end = frame_idx + 1
            traj_start = max(0, traj_end - 50)
            trajectory = np.array([h_no_gp['positions'][j][i] for j in range(traj_start, traj_end)])
            if len(trajectory) > 1:
                ax5.plot(trajectory[:, 0], trajectory[:, 1], color=colors[i], alpha=0.6, linewidth=2)
            ax5.scatter(positions_no_gp[i, 0], positions_no_gp[i, 1], c=[colors[i]], s=150,
                        marker='o', edgecolors='black', linewidths=2, zorder=15)

        ax5.set_title('Without GP Prediction', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X (m)')
        ax5.set_ylabel('Y (m)')
        ax5.set_xlim(0, 100)
        ax5.set_ylim(0, 100)
        ax5.set_aspect('equal')

        # 图6: 预测误差统计
        ax6 = axes[1, 2]
        pred_errors = np.array(h['prediction_error'][:frame_idx + 1])
        time_arr = np.array(h['time'][:frame_idx + 1])

        ax6.plot(time_arr, pred_errors, 'purple', linewidth=2, label='Prediction Error')
        ax6.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Threshold (10m)')
        ax6.fill_between(time_arr, pred_errors, alpha=0.3, color='purple')

        # 标注当前误差
        if len(pred_errors) > 0:
            current_error = pred_errors[-1]
            ax6.scatter(time_arr[-1], current_error, c='red', s=100, zorder=10)
            ax6.annotate('Current: %.1fm' % current_error, (time_arr[-1], current_error),
                         textcoords="offset points", xytext=(-50, 10), fontsize=10, fontweight='bold')

        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Prediction Error (m)')
        ax6.set_title('GP Prediction Error Over Time', fontsize=12, fontweight='bold')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, self.total_time)

        plt.suptitle('Field Prediction Comparison (t=%.1fs)' % current_time,
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "field_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved: %s" % save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_keyframes(self, num_frames=6):
        """生成关键帧"""
        keyframes_dir = os.path.join(self.output_dir, "keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)

        total = len(self.sim_gp.history['time'])
        indices = np.linspace(0, total - 1, num_frames, dtype=int)

        print("Generating %d keyframes..." % num_frames)

        for i, idx in enumerate(indices):
            save_path = os.path.join(keyframes_dir, "frame_%03d.png" % i)
            self.plot_field_comparison(frame_idx=idx, save_path=save_path, show=False)

        print("Keyframes saved to: %s" % keyframes_dir)

    def create_animation(self, save_path=None, fps=10):
        """创建动画：显示真实场、预测场、无人机"""
        print("Creating prediction visualization animation...")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        h = self.sim_gp.history
        num_frames = len(h['time'])
        vmin, vmax = 0, 2.5

        print("Total frames: %d" % num_frames)

        def animate(frame_idx):
            for ax in axes:
                ax.clear()

            current_time = h['time'][frame_idx]
            current_field = h['current_field'][frame_idx]
            predicted_field = h['predicted_field'][frame_idx]

            current_hotspot = h['hotspot_pos'][frame_idx]
            predicted_hotspot = h['predicted_hotspot_pos'][frame_idx]
            future_hotspot = h['future_hotspot_pos'][frame_idx]

            positions_gp = h['positions'][frame_idx]
            positions_no_gp = self.sim_no_gp.history['positions'][frame_idx]

            # 图1: 当前场 + 两组无人机
            ax1 = axes[0]
            ax1.imshow(current_field, extent=[0, 100, 0, 100], origin='lower',
                       cmap='YlOrRd', alpha=0.7, vmin=vmin, vmax=vmax)
            ax1.scatter(current_hotspot[0], current_hotspot[1], c='red', s=200, marker='*',
                        edgecolors='darkred', linewidths=2, zorder=20)

            # GP无人机（蓝色边框）
            for i in range(self.num_agents):
                ax1.scatter(positions_gp[i, 0], positions_gp[i, 1], c=[colors[i]], s=120,
                            marker='o', edgecolors='blue', linewidths=3, zorder=15)

            # NoGP无人机（红色边框）
            for i in range(self.num_agents):
                ax1.scatter(positions_no_gp[i, 0], positions_no_gp[i, 1], c=[colors[i]], s=120,
                            marker='s', edgecolors='red', linewidths=3, zorder=14, alpha=0.7)

            dist_gp = h['hotspot_distance'][frame_idx]
            dist_no_gp = self.sim_no_gp.history['hotspot_distance'][frame_idx]
            ax1.set_title('Current Field | GP(o):%.1fm NoGP(s):%.1fm' % (dist_gp, dist_no_gp),
                          fontsize=10, fontweight='bold')
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 100)
            ax1.set_aspect('equal')

            # 图2: GP预测场
            ax2 = axes[1]
            if predicted_field is not None:
                ax2.imshow(predicted_field, extent=[0, 100, 0, 100], origin='lower',
                           cmap='YlOrRd', alpha=0.7, vmin=vmin, vmax=vmax)
                if predicted_hotspot is not None:
                    ax2.scatter(predicted_hotspot[0], predicted_hotspot[1], c='blue', s=200, marker='*',
                                edgecolors='darkblue', linewidths=2, zorder=20, label='Predicted')
                if future_hotspot is not None:
                    ax2.scatter(future_hotspot[0], future_hotspot[1], c='green', s=150, marker='x',
                                linewidths=3, zorder=19, label='Actual Future')
                    # 预测误差线
                    if predicted_hotspot is not None:
                        ax2.plot([predicted_hotspot[0], future_hotspot[0]],
                                 [predicted_hotspot[1], future_hotspot[1]],
                                 'r--', linewidth=2, alpha=0.8)
                ax2.legend(loc='upper right', fontsize=8)
                pred_err = h['prediction_error'][frame_idx]
                ax2.set_title('GP Prediction | Error: %.1fm' % pred_err, fontsize=10, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'GP Training...', ha='center', va='center',
                         transform=ax2.transAxes, fontsize=14)
                ax2.set_title('GP Prediction (Not Ready)', fontsize=10, fontweight='bold')
            ax2.set_xlim(0, 100)
            ax2.set_ylim(0, 100)
            ax2.set_aspect('equal')

            # 图3: 预测误差曲线
            ax3 = axes[2]
            time_arr = np.array(h['time'][:frame_idx + 1])
            pred_errors = np.array(h['prediction_error'][:frame_idx + 1])

            ax3.plot(time_arr, pred_errors, 'purple', linewidth=2)
            ax3.fill_between(time_arr, pred_errors, alpha=0.3, color='purple')
            ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.7)

            if len(pred_errors) > 0:
                ax3.scatter(time_arr[-1], pred_errors[-1], c='red', s=80, zorder=10)

            ax3.set_xlim(0, self.total_time)
            ax3.set_ylim(0, max(50, np.max(pred_errors) * 1.2) if len(pred_errors) > 0 else 50)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Pred Error (m)')
            ax3.set_title('Prediction Error', fontsize=10, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            plt.suptitle('t = %.1fs' % current_time, fontsize=12, fontweight='bold')
            plt.tight_layout()

            if frame_idx % 20 == 0:
                print("  Rendering frame %d/%d" % (frame_idx, num_frames))

            return axes

        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000 // fps, blit=False)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "prediction_animation.gif")

        print("Saving to %s..." % save_path)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        print("Animation saved: %s" % save_path)

        plt.close(fig)

    def print_statistics(self):
        """打印统计"""
        h_gp = self.sim_gp.history
        h_no_gp = self.sim_no_gp.history

        half = len(h_gp['time']) // 2

        print("\n" + "=" * 60)
        print("PREDICTION VISUALIZATION RESULTS")
        print("=" * 60)

        # 预测误差
        pred_errors = np.array(h_gp['prediction_error'][half:])
        valid_errors = pred_errors[pred_errors > 0]

        if len(valid_errors) > 0:
            print("GP Prediction Error (after training):")
            print("  Average: %.1f m" % np.mean(valid_errors))
            print("  Min:     %.1f m" % np.min(valid_errors))
            print("  Max:     %.1f m" % np.max(valid_errors))

        print("-" * 60)

        # 性能对比
        dist_gp = np.mean(h_gp['hotspot_distance'][half:])
        dist_no_gp = np.mean(h_no_gp['hotspot_distance'][half:])

        print("Hotspot Tracking Distance:")
        print("  With GP:    %.1f m" % dist_gp)
        print("  Without GP: %.1f m" % dist_no_gp)

        improvement = (dist_no_gp - dist_gp) / dist_no_gp * 100
        print("  Improvement: %.1f%%" % improvement)

        print("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Prediction Visualization')
    parser.add_argument('--agents', type=int, default=3)
    parser.add_argument('--time', type=float, default=60.0)
    parser.add_argument('--motion', type=str, default='circular', choices=['linear', 'circular'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keyframes', type=int, default=6)
    parser.add_argument('--no-animation', action='store_true')
    parser.add_argument('--fps', type=int, default=10)

    args = parser.parse_args()

    print("=" * 60)
    print("Prediction Field Visualization Experiment")
    print("=" * 60)
    print("UAVs: %d" % args.agents)
    print("Time: %.1fs" % args.time)
    print("Motion: %s" % args.motion)
    print("=" * 60)

    exp = PredictionComparisonExperiment(
        num_agents=args.agents,
        total_time=args.time,
        motion_type=args.motion,
        seed=args.seed
    )

    exp.run(verbose=True)
    exp.print_statistics()

    print("\nGenerating visualizations...")

    # 1. 最终帧对比图
    print("\n[1/3] Final frame comparison...")
    exp.plot_field_comparison(frame_idx=-1, show=False)

    # 2. 关键帧
    print("\n[2/3] Keyframes...")
    exp.plot_keyframes(num_frames=args.keyframes)

    # 3. 动画
    if not args.no_animation:
        print("\n[3/3] Creating animation...")
        exp.create_animation(fps=args.fps)

    print("\nOutputs saved to: %s" % exp.output_dir)

    # 显示最终结果
    exp.plot_field_comparison(frame_idx=-1, show=True)


if __name__ == "__main__":
    main()