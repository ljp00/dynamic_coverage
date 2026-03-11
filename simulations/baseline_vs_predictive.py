"""
对比实验：基线（反应式局部）vs 改进（GP预测）
目标：对动态敏感度场保持高覆盖率
"""
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage.coverage_controllers import ReactiveLocalController, PredictiveGPController
from src.safety.cbf import DistributedCBF


class DynamicSensitivityField:
    """
    动态敏感度场
    热点做圆周运动，模拟动态变化的覆盖需求
    """

    def __init__(self, domain=(0, 100, 0, 100), resolution=50,
                 motion_type="circular", seed=None):
        self.domain = domain
        self.resolution = resolution
        self.motion_type = motion_type
        self.time = 0.0

        if seed is not None:
            np.random.seed(seed)

        # 网格
        self.x_grid = np.linspace(domain[0], domain[1], resolution)
        self.y_grid = np.linspace(domain[2], domain[3], resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

        # 热点参数
        self.origin = np.array([50.0, 50.0])
        self.amplitude = 25.0
        self.omega = 0.2  # 角速度
        self.intensity = 2.0
        self.spread = 15.0

        self.center = self.origin.copy()

    def update(self, dt):
        """更新敏感度场（热点移动）"""
        self.time += dt

        if self.motion_type == "circular":
            angle = self.omega * self.time
            self.center = self.origin + self.amplitude * np.array([
                np.cos(angle), np.sin(angle)
            ])
        elif self.motion_type == "linear":
            self.center = self.origin + np.array([
                self.amplitude * np.sin(self.omega * self.time), 0
            ])
        elif self.motion_type == "figure8":
            t = self.omega * self.time
            self.center = self.origin + np.array([
                self.amplitude * np.sin(t),
                self.amplitude * np.sin(2 * t) / 2
            ])

    def get_density(self, positions):
        """
        获取指定位置的敏感度值
        这是智能体通过传感器可以测量到的
        """
        positions = np.atleast_2d(positions)
        dist_sq = np.sum((positions - self.center) ** 2, axis=1)
        density = self.intensity * np.exp(-dist_sq / (2 * self.spread ** 2))
        density += 0.1  # 基础敏感度
        return density

    def get_field_grid(self):
        """获取完整敏感度场（仅用于可视化和评估）"""
        positions = np.column_stack([self.X.ravel(), self.Y.ravel()])
        density = self.get_density(positions)
        return density.reshape(self.resolution, self.resolution)

    def get_hotspot_position(self):
        """获取真实热点位置（仅用于评估）"""
        return self.center.copy()


class CoverageMetrics:
    """
    覆盖率评估指标
    """

    def __init__(self, domain=(0, 100, 0, 100), resolution=50, sensing_radius=15.0):
        self.domain = domain
        self.resolution = resolution
        self.sensing_radius = sensing_radius

        # 评估网格
        self.x_grid = np.linspace(domain[0], domain[1], resolution)
        self.y_grid = np.linspace(domain[2], domain[3], resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.grid_points = np.column_stack([self.X.ravel(), self.Y.ravel()])

    def compute_weighted_coverage(self, positions: np.ndarray, field: np.ndarray) -> float:
        """
        计算加权覆盖率

        覆盖率 = Σ(被覆盖点的敏感度) / Σ(所有点的敏感度)
        """
        # 判断每个网格点是否被覆盖
        covered = np.zeros(len(self.grid_points), dtype=bool)

        for pos in positions:
            distances = np.linalg.norm(self.grid_points - pos, axis=1)
            covered |= (distances <= self.sensing_radius)

        # 计算加权覆盖率
        field_flat = field.ravel()
        total_sensitivity = np.sum(field_flat)
        covered_sensitivity = np.sum(field_flat[covered])

        if total_sensitivity > 0:
            return covered_sensitivity / total_sensitivity
        return 0.0

    def compute_coverage_cost(self, positions: np.ndarray, field: np.ndarray) -> float:
        """
        计算覆盖代价（越低越好）

        代价 = Σ s(q) * min_i ||q - p_i||^2
        """
        field_flat = field.ravel()

        # 计算每个网格点到最近智能体的距离
        min_distances_sq = np.full(len(self.grid_points), np.inf)

        for pos in positions:
            distances_sq = np.sum((self.grid_points - pos) ** 2, axis=1)
            min_distances_sq = np.minimum(min_distances_sq, distances_sq)

        # 加权代价
        cost = np.sum(field_flat * min_distances_sq)

        return cost

    def compute_hotspot_distance(self, positions: np.ndarray, hotspot: np.ndarray) -> float:
        """
        计算最近智能体到热点的距离
        """
        distances = np.linalg.norm(positions - hotspot, axis=1)
        return np.min(distances)


class BaselineSimulator:
    """
    基线算法仿真器
    使用反应式局部控制
    """

    def __init__(self, field, num_agents=3, seed=42):
        self.field = field
        self.domain = field.domain
        self.num_agents = num_agents

        np.random.seed(seed)
        self.positions = np.random.uniform(
            [self.domain[0] + 20, self.domain[2] + 20],
            [self.domain[1] - 20, self.domain[3] - 20],
            (num_agents, 2)
        )

        self.controller = ReactiveLocalController(
            domain=self.domain,
            max_velocity=5.0,
            sensing_radius=15.0,
            gradient_gain=2.0,
            local_max_gain=1.5
        )

        for i in range(num_agents):
            self.controller.initialize_agent(i)

        self.cbf = DistributedCBF(safe_distance=3.0, gamma=1.0, domain=self.domain)
        self.metrics = CoverageMetrics(self.domain, sensing_radius=15.0)

        self.time = 0.0
        self.history = {
            'time': [],
            'positions': [],
            'coverage_rate': [],
            'coverage_cost': [],
            'hotspot_distance': []
        }

    def step(self, dt):
        velocities = np.zeros_like(self.positions)

        for i in range(self.num_agents):
            other_positions = np.delete(self.positions, i, axis=0)

            velocity, _ = self.controller.compute_control(
                agent_id=i,
                position=self.positions[i],
                field_func=self.field.get_density,
                current_time=self.time,
                other_positions=other_positions
            )
            velocities[i] = velocity

        # CBF安全过滤
        velocities = self.cbf.filter_control(self.positions, velocities)

        # 更新位置
        self.positions += velocities * dt

        # 记录
        self._record()
        self.time += dt

    def _record(self):
        field = self.field.get_field_grid()
        hotspot = self.field.get_hotspot_position()

        coverage_rate = self.metrics.compute_weighted_coverage(self.positions, field)
        coverage_cost = self.metrics.compute_coverage_cost(self.positions, field)
        hotspot_dist = self.metrics.compute_hotspot_distance(self.positions, hotspot)

        self.history['time'].append(self.time)
        self.history['positions'].append(self.positions.copy())
        self.history['coverage_rate'].append(coverage_rate)
        self.history['coverage_cost'].append(coverage_cost)
        self.history['hotspot_distance'].append(hotspot_dist)


class PredictiveSimulator:
    """改进算法仿真器"""

    def __init__(self, field, num_agents=3, seed=42, learning_duration=15.0):
        self.field = field
        self.domain = field.domain
        self.num_agents = num_agents

        np.random.seed(seed)
        self.positions = np.random.uniform(
            [self.domain[0] + 20, self.domain[2] + 20],
            [self.domain[1] - 20, self.domain[3] - 20],
            (num_agents, 2)
        )

        self.controller = PredictiveGPController(
            domain=self.domain,
            max_velocity=5.0,
            sensing_radius=15.0,
            prediction_horizon=0.5,
            learning_duration=learning_duration,
            max_prediction_weight=0.5  # 正确的参数名
        )

        for i in range(num_agents):
            self.controller.initialize_agent(i)

        self.cbf = DistributedCBF(safe_distance=3.0, gamma=1.0, domain=self.domain)
        self.metrics = CoverageMetrics(self.domain, sensing_radius=15.0)

        self.time = 0.0
        self.history = {
            'time': [],
            'positions': [],
            'coverage_rate': [],
            'coverage_cost': [],
            'hotspot_distance': [],
            'mode': [],
            'predicted_hotspot': [],
            'prediction_confidence': [],
            'prediction_weight': []
        }

    def step(self, dt):
        velocities = np.zeros_like(self.positions)
        modes = []
        predicted_hotspots = []
        confidences = []
        weights = []

        for i in range(self.num_agents):
            other_positions = np.delete(self.positions, i, axis=0)

            velocity, info = self.controller.compute_control(
                agent_id=i,
                position=self.positions[i],
                field_func=self.field.get_density,
                current_time=self.time,
                other_positions=other_positions,
                n_agents=self.num_agents
            )
            velocities[i] = velocity
            modes.append(info.get('mode', 'unknown'))

            pred_hotspot = info.get('predicted_hotspot', None)
            predicted_hotspots.append(pred_hotspot)
            confidences.append(info.get('prediction_confidence', 0))
            weights.append(info.get('prediction_weight', 0))

        velocities = self.cbf.filter_control(self.positions, velocities)
        self.positions += velocities * dt
        self._record(modes, predicted_hotspots, confidences, weights)
        self.time += dt

    def _record(self, modes, predicted_hotspots, confidences, weights):
        field = self.field.get_field_grid()
        hotspot = self.field.get_hotspot_position()

        coverage_rate = self.metrics.compute_weighted_coverage(self.positions, field)
        coverage_cost = self.metrics.compute_coverage_cost(self.positions, field)
        hotspot_dist = self.metrics.compute_hotspot_distance(self.positions, hotspot)

        self.history['time'].append(self.time)
        self.history['positions'].append(self.positions.copy())
        self.history['coverage_rate'].append(coverage_rate)
        self.history['coverage_cost'].append(coverage_cost)
        self.history['hotspot_distance'].append(hotspot_dist)
        self.history['mode'].append(modes[0] if modes else 'unknown')
        self.history['predicted_hotspot'].append(predicted_hotspots[0] if predicted_hotspots else None)
        self.history['prediction_confidence'].append(np.mean(confidences) if confidences else 0)
        self.history['prediction_weight'].append(np.mean(weights) if weights else 0)


class BaselineVsPredictiveExperiment:
    """对比实验"""

    def __init__(self, num_agents=3, total_time=100.0, dt=0.1,
                 motion_type="circular", learning_duration=15.0, seed=42):
        self.num_agents = num_agents
        self.total_time = total_time
        self.dt = dt
        self.motion_type = motion_type
        self.learning_duration = learning_duration
        self.seed = seed
        self.domain = (0, 100, 0, 100)

        self.field = DynamicSensitivityField(
            domain=self.domain, resolution=50, motion_type=motion_type, seed=seed
        )

        self.sim_baseline = BaselineSimulator(self.field, num_agents, seed=seed)
        self.sim_predictive = PredictiveSimulator(
            self.field, num_agents, seed=seed, learning_duration=learning_duration
        )

        self.time = 0.0
        self.field_history = []
        self.hotspot_history = []

        self.output_dir = "output/baseline_vs_predictive"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, verbose=True):
        num_steps = int(self.total_time / self.dt)

        if verbose:
            print("=" * 75)
            print("BASELINE vs PREDICTIVE COVERAGE CONTROL")
            print("=" * 75)
            print("")
            print("Baseline (Reactive Local):")
            print("  - Only local sensing (no future prediction)")
            print("  - Gradient ascent + local maximum tracking")
            print("")
            print("Predictive (GP-based):")
            print("  - Initial learning phase: %.0fs" % self.learning_duration)
            print("  - GP prediction of future sensitivity field")
            print("  - Proactive deployment to predicted high-value regions")
            print("")
            print("Settings:")
            print("  - Agents: %d" % self.num_agents)
            print("  - Total time: %.0fs" % self.total_time)
            print("  - Motion type: %s" % self.motion_type)
            print("-" * 75)

        for i in range(num_steps):
            # 更新敏感度场
            self.field.update(self.dt)

            # 两个仿真器同步运行
            self.sim_baseline.step(self.dt)
            self.sim_predictive.step(self.dt)

            # 记录场历史
            if i % 2 == 0:
                self.field_history.append(self.field.get_field_grid().copy())
                self.hotspot_history.append(self.field.get_hotspot_position().copy())

            self.time += self.dt

            if verbose and i % 100 == 0:
                cov_base = self.sim_baseline.history['coverage_rate'][-1]
                cov_pred = self.sim_predictive.history['coverage_rate'][-1]
                dist_base = self.sim_baseline.history['hotspot_distance'][-1]
                dist_pred = self.sim_predictive.history['hotspot_distance'][-1]
                mode = self.sim_predictive.history['mode'][-1]

                print("Step %4d: Baseline(Cov=%.1f%%, Dist=%.1fm) | Predictive(Cov=%.1f%%, Dist=%.1fm, Mode=%s)" %
                      (i, cov_base * 100, dist_base, cov_pred * 100, dist_pred, mode))

        if verbose:
            print("-" * 75)
            print("Simulation complete!")

    def print_statistics(self):
        h_base = self.sim_baseline.history
        h_pred = self.sim_predictive.history

        # 分阶段统计
        learning_end_idx = int(self.learning_duration / self.dt)

        print("\n" + "=" * 75)
        print("COMPARISON RESULTS")
        print("=" * 75)

        # 学习阶段统计
        print("\n【Learning Phase (0-%.0fs)】" % self.learning_duration)
        print("%-30s %15s %15s" % ("Metric", "Baseline", "Predictive"))
        print("-" * 60)

        cov_base_learn = np.mean(h_base['coverage_rate'][:learning_end_idx])
        cov_pred_learn = np.mean(h_pred['coverage_rate'][:learning_end_idx])
        print("%-30s %14.1f%% %14.1f%%" % ("Avg Coverage Rate", cov_base_learn * 100, cov_pred_learn * 100))

        # 运行阶段统计（学习后）
        print("\n【Deployment Phase (%.0fs-%.0fs)】" % (self.learning_duration, self.total_time))
        print("%-30s %15s %15s" % ("Metric", "Baseline", "Predictive"))
        print("-" * 60)

        cov_base = np.mean(h_base['coverage_rate'][learning_end_idx:])
        cov_pred = np.mean(h_pred['coverage_rate'][learning_end_idx:])
        print("%-30s %14.1f%% %14.1f%%" % ("Avg Coverage Rate", cov_base * 100, cov_pred * 100))

        cov_base_std = np.std(h_base['coverage_rate'][learning_end_idx:])
        cov_pred_std = np.std(h_pred['coverage_rate'][learning_end_idx:])
        print("%-30s %14.1f%% %14.1f%%" % ("Coverage Stability (Std)", cov_base_std * 100, cov_pred_std * 100))

        dist_base = np.mean(h_base['hotspot_distance'][learning_end_idx:])
        dist_pred = np.mean(h_pred['hotspot_distance'][learning_end_idx:])
        print("%-30s %15.1f %15.1f" % ("Avg Hotspot Distance (m)", dist_base, dist_pred))

        cost_base = np.mean(h_base['coverage_cost'][learning_end_idx:])
        cost_pred = np.mean(h_pred['coverage_cost'][learning_end_idx:])
        print("%-30s %15.0f %15.0f" % ("Avg Coverage Cost", cost_base, cost_pred))

        print("-" * 60)

        # 改善率
        cov_improvement = (cov_pred - cov_base) / cov_base * 100 if cov_base > 0 else 0
        dist_improvement = (dist_base - dist_pred) / dist_base * 100 if dist_base > 0 else 0
        cost_improvement = (cost_base - cost_pred) / cost_base * 100 if cost_base > 0 else 0

        print("\n【Improvement (Predictive vs Baseline)】")
        print("%-30s %+14.1f%%" % ("Coverage Rate Improvement", cov_improvement))
        print("%-30s %+14.1f%%" % ("Hotspot Distance Improvement", dist_improvement))
        print("%-30s %+14.1f%%" % ("Coverage Cost Improvement", cost_improvement))

        print("=" * 75)

        # 结论
        print("\nCONCLUSION:")
        if cov_improvement > 5:
            print("  ✓ Predictive control significantly outperforms baseline!")
            print("  ✓ GP prediction enables proactive deployment")
        elif cov_improvement > 0:
            print("  ○ Predictive control shows moderate improvement")
        else:
            print("  ✗ Baseline performs comparably or better")
            print("    Consider: longer learning time, different parameters")

    def plot_comparison(self, save_path=None, show=True):
        """绘制对比曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        h_base = self.sim_baseline.history
        h_pred = self.sim_predictive.history
        time = np.array(h_base['time'])

        learning_time = self.learning_duration

        # 图1: 覆盖率对比
        ax1 = axes[0, 0]
        ax1.plot(time, np.array(h_base['coverage_rate']) * 100, 'r-',
                 linewidth=2, label='Baseline (Reactive)', alpha=0.8)
        ax1.plot(time, np.array(h_pred['coverage_rate']) * 100, 'b-',
                 linewidth=2, label='Predictive (GP)')
        ax1.axvline(x=learning_time, color='green', linestyle='--',
                    linewidth=2, label='Learning ends')
        ax1.fill_between([0, learning_time], 0, 100, alpha=0.1, color='green')
        ax1.set_ylabel('Weighted Coverage Rate (%)')
        ax1.set_title('Coverage Rate Comparison', fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        # 图2: 热点距离对比
        ax2 = axes[0, 1]
        ax2.plot(time, h_base['hotspot_distance'], 'r-',
                 linewidth=2, label='Baseline', alpha=0.8)
        ax2.plot(time, h_pred['hotspot_distance'], 'b-',
                 linewidth=2, label='Predictive')
        ax2.axvline(x=learning_time, color='green', linestyle='--', linewidth=2)
        ax2.set_ylabel('Min Distance to Hotspot (m)')
        ax2.set_title('Hotspot Tracking Performance', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图3: 覆盖代价对比
        ax3 = axes[1, 0]
        ax3.plot(time, h_base['coverage_cost'], 'r-',
                 linewidth=2, label='Baseline', alpha=0.8)
        ax3.plot(time, h_pred['coverage_cost'], 'b-',
                 linewidth=2, label='Predictive')
        ax3.axvline(x=learning_time, color='green', linestyle='--', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Coverage Cost')
        ax3.set_title('Coverage Cost (Lower is Better)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # 图4: 性能统计柱状图
        ax4 = axes[1, 1]

        # 学习后的统计
        learn_idx = int(learning_time / self.dt)
        metrics = ['Coverage\nRate (%)', 'Hotspot\nDist (m)', 'Cost\n(×10³)']
        base_vals = [
            np.mean(h_base['coverage_rate'][learn_idx:]) * 100,
            np.mean(h_base['hotspot_distance'][learn_idx:]),
            np.mean(h_base['coverage_cost'][learn_idx:]) / 1000
        ]
        pred_vals = [
            np.mean(h_pred['coverage_rate'][learn_idx:]) * 100,
            np.mean(h_pred['hotspot_distance'][learn_idx:]),
            np.mean(h_pred['coverage_cost'][learn_idx:]) / 1000
        ]

        x = np.arange(len(metrics))
        width = 0.35

        bars1 = ax4.bar(x - width / 2, base_vals, width, label='Baseline', color='red', alpha=0.7)
        bars2 = ax4.bar(x + width / 2, pred_vals, width, label='Predictive', color='blue', alpha=0.7)

        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Summary (After Learning)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 标注数值
        for bar, val in zip(bars1, base_vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     '%.1f' % val, ha='center', va='bottom', fontsize=9, color='red')
        for bar, val in zip(bars2, pred_vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     '%.1f' % val, ha='center', va='bottom', fontsize=9, color='blue')

        plt.suptitle('Baseline vs Predictive Coverage Control (%s motion)' % self.motion_type,
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "comparison_curves.png")
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

        data_idx = min(frame_idx * 2, len(self.sim_baseline.history['time']) - 1)

        field = self.field_history[frame_idx]
        hotspot = self.hotspot_history[frame_idx]
        current_time = self.sim_baseline.history['time'][data_idx]

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        sims = [
            (self.sim_baseline, "Baseline (Reactive Local)"),
            (self.sim_predictive, "Predictive (GP-based)")
        ]

        for ax_idx, (sim, title) in enumerate(sims):
            ax = axes[ax_idx]

            # 敏感度场
            ax.imshow(field, extent=[0, 100, 0, 100], origin='lower',
                      cmap='YlOrRd', alpha=0.7, vmin=0, vmax=2.5)

            # 热点
            ax.scatter(hotspot[0], hotspot[1], c='red', s=200, marker='*',
                       edgecolors='darkred', linewidths=2, zorder=20, label='Hotspot')

            # 热点轨迹（圆）
            if self.motion_type == "circular":
                theta = np.linspace(0, 2 * np.pi, 100)
                circle_x = 50 + 25 * np.cos(theta)
                circle_y = 50 + 25 * np.sin(theta)
                ax.plot(circle_x, circle_y, 'r--', alpha=0.3, linewidth=1)

            # 预测热点（仅predictive）
            if ax_idx == 1 and data_idx < len(sim.history['predicted_hotspot']):
                pred_hotspot = sim.history['predicted_hotspot'][data_idx]
                if pred_hotspot is not None:
                    ax.scatter(pred_hotspot[0], pred_hotspot[1], c='blue', s=150, marker='x',
                               linewidths=3, zorder=19, label='Predicted')

            # 智能体轨迹和位置
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
                        ax.plot(trajectory[k:k + 2, 0], trajectory[k:k + 2, 1],
                                color=colors[i], alpha=alphas[k], linewidth=2)

                # 当前位置
                ax.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=150,
                           marker='o', edgecolors='black', linewidths=2, zorder=15)

                # 感知范围
                circle = plt.Circle(positions[i], 15, fill=False,
                                    color=colors[i], linestyle=':', alpha=0.5)
                ax.add_patch(circle)

            # 标题信息
            cov = sim.history['coverage_rate'][data_idx] * 100
            dist = sim.history['hotspot_distance'][data_idx]

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('%s\nt=%.1fs, Coverage=%.1f%%, Dist=%.1fm' % (title, current_time, cov, dist),
                         fontsize=11, fontweight='bold')
            ax.set_aspect('equal')
            ax.legend(loc='upper right', fontsize=8)
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
        """创建对比动画"""
        print("Creating animation...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
        num_frames = len(self.field_history)

        print("Total frames: %d" % num_frames)

        def animate(frame_idx):
            for ax in axes:
                ax.clear()

            data_idx = min(frame_idx * 2, len(self.sim_baseline.history['time']) - 1)
            field = self.field_history[frame_idx]
            hotspot = self.hotspot_history[frame_idx]
            current_time = self.sim_baseline.history['time'][data_idx]

            sims = [
                (self.sim_baseline, "Baseline (Reactive)"),
                (self.sim_predictive, "Predictive (GP)")
            ]

            for ax_idx, (sim, title) in enumerate(sims):
                ax = axes[ax_idx]

                # 敏感度场
                ax.imshow(field, extent=[0, 100, 0, 100], origin='lower',
                          cmap='YlOrRd', alpha=0.6, vmin=0, vmax=2.5)

                # 热点
                ax.scatter(hotspot[0], hotspot[1], c='red', s=200, marker='*',
                           edgecolors='darkred', linewidths=2, zorder=20)

                # 热点轨迹
                if self.motion_type == "circular":
                    theta = np.linspace(0, 2 * np.pi, 100)
                    ax.plot(50 + 25 * np.cos(theta), 50 + 25 * np.sin(theta),
                            'r--', alpha=0.3, linewidth=1)

                # 预测热点
                if ax_idx == 1 and data_idx < len(sim.history['predicted_hotspot']):
                    pred = sim.history['predicted_hotspot'][data_idx]
                    if pred is not None:
                        ax.scatter(pred[0], pred[1], c='blue', s=100, marker='x',
                                   linewidths=2, zorder=19, alpha=0.8)

                positions = sim.history['positions'][data_idx]

                for i in range(self.num_agents):
                    # 轨迹
                    traj_end = data_idx + 1
                    traj_start = max(0, traj_end - 40)
                    trajectory = np.array([sim.history['positions'][j][i]
                                           for j in range(traj_start, traj_end)])
                    if len(trajectory) > 1:
                        ax.plot(trajectory[:, 0], trajectory[:, 1],
                                color=colors[i], alpha=0.5, linewidth=1.5)

                    ax.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=120,
                               marker='o', edgecolors='black', linewidths=2, zorder=15)

                    # 感知范围
                    circle = plt.Circle(positions[i], 15, fill=False,
                                        color=colors[i], linestyle=':', alpha=0.3)
                    ax.add_patch(circle)

                cov = sim.history['coverage_rate'][data_idx] * 100
                dist = sim.history['hotspot_distance'][data_idx]

                # 模式标识
                mode_str = ""
                if ax_idx == 1 and data_idx < len(sim.history['mode']):
                    mode = sim.history['mode'][data_idx]
                    mode_str = " [%s]" % mode

                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.set_title('%s%s\nCov=%.1f%%, Dist=%.1fm' % (title, mode_str, cov, dist),
                             fontsize=10, fontweight='bold')
                ax.set_aspect('equal')

            # 学习阶段指示
            if current_time < self.learning_duration:
                axes[1].text(50, 95, 'LEARNING PHASE', ha='center', va='top',
                             fontsize=12, fontweight='bold', color='green',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle('t = %.1fs' % current_time, fontsize=12, fontweight='bold')
            plt.tight_layout()

            if frame_idx % 30 == 0:
                print("  Frame %d/%d" % (frame_idx, num_frames))

            return axes

        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000 // fps, blit=False)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "comparison_animation.gif")

        print("Saving to %s..." % save_path)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        print("Animation saved: %s" % save_path)

        plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Baseline vs Predictive Coverage Control')
    parser.add_argument('--agents', type=int, default=3, help='Number of agents')
    parser.add_argument('--time', type=float, default=80.0, help='Total simulation time')
    parser.add_argument('--learning', type=float, default=10.0, help='Learning duration for GP')
    parser.add_argument('--motion', type=str, default='circular',
                        choices=['circular', 'linear', 'figure8'],
                        help='Hotspot motion type')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-animation', action='store_true', help='Skip animation')
    parser.add_argument('--fps', type=int, default=10, help='Animation FPS')

    args = parser.parse_args()

    exp = BaselineVsPredictiveExperiment(
        num_agents=args.agents,
        total_time=args.time,
        learning_duration=args.learning,
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

    print("\nAll outputs saved to: %s" % exp.output_dir)
    exp.plot_comparison(show=True)


# 注意：这行必须在文件最外层，没有缩进
if __name__ == "__main__":
    main()