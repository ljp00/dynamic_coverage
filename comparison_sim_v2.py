"""
对比实验：GP预测 vs 无GP预测 的覆盖控制效果
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coverage. voronoi import WeightedVoronoi
from src.coverage.lloyd_controller import LloydController
from src.safety.cbf import DistributedCBF
from src.agents.uav import UAVSwarm
from src.prediction.gp_predictor import SpatioTemporalGP


class SimpleMovingHotspot:
    """简单的单热点移动场"""

    def __init__(self, domain=(0, 100, 0, 100), resolution=50, motion_type="linear", seed=None):
        self.domain = domain
        self. resolution = resolution
        self.motion_type = motion_type
        self.time = 0.0

        if seed is not None:
            np.random. seed(seed)

        self.x_grid = np.linspace(domain[0], domain[1], resolution)
        self.y_grid = np.linspace(domain[2], domain[3], resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

        self.intensity = 2.0
        self.spread = 15.0

        cx = (domain[0] + domain[1]) / 2
        cy = (domain[2] + domain[3]) / 2

        if motion_type == "linear":
            self.origin = np.array([cx, cy])
            self.amplitude = np.array([30.0, 0.0])
            self.velocity = 0.3
        elif motion_type == "circular":
            self.origin = np.array([cx, cy])
            self.amplitude = np. array([25.0, 25.0])
            self.velocity = 0.2
        elif motion_type == "diagonal":
            self.origin = np.array([cx, cy])
            self.amplitude = np.array([25.0, 25.0])
            self.velocity = 0.25
        else:
            self.origin = np.array([cx, cy])
            self.amplitude = np.array([30.0, 0.0])
            self.velocity = 0.3

        self.center = self.origin. copy()

    def update(self, dt):
        self.time += dt

        if self.motion_type == "linear":
            self.center = self.origin + self.amplitude * np.sin(self.velocity * self.time)
        elif self.motion_type == "circular":
            angle = self.velocity * self.time
            self.center = self. origin + np.array([
                self.amplitude[0] * np.cos(angle),
                self.amplitude[1] * np.sin(angle)
            ])
        elif self.motion_type == "diagonal":
            self.center = self.origin + self.amplitude * np.sin(self.velocity * self.time)

        margin = 5
        self.center[0] = np.clip(self.center[0], self.domain[0] + margin, self.domain[1] - margin)
        self.center[1] = np.clip(self.center[1], self. domain[2] + margin, self.domain[3] - margin)

    def get_density(self, positions):
        positions = np.atleast_2d(positions)
        dist_sq = np.sum((positions - self.center) ** 2, axis=1)
        density = self.intensity * np.exp(-dist_sq / (2 * self.spread ** 2))
        density += 0.1
        return density

    def get_field_grid(self):
        positions = np.column_stack([self.X. ravel(), self.Y.ravel()])
        density = self.get_density(positions)
        return density.reshape(self.resolution, self.resolution)

    def get_hotspot_position(self):
        return self.center.copy()


class SingleSimulator:
    """单组无人机仿真器"""

    def __init__(self, field, num_agents=5, use_gp=True, name="Simulator"):
        self.field = field
        self.domain = field.domain
        self. num_agents = num_agents
        self.use_gp = use_gp
        self. name = name

        self.swarm = UAVSwarm(num_agents, self.domain, max_velocity=5.0, sensing_radius=15.0)
        self.controller = LloydController(self. domain, gain=1.5, max_velocity=5.0, resolution=50)
        self.voronoi = WeightedVoronoi(self.domain, resolution=50)
        self.cbf = DistributedCBF(safe_distance=3.0, gamma=1.0, domain=self.domain, communication_radius=30.0)

        if use_gp:
            self.gp = SpatioTemporalGP(length_scale_space=15.0, length_scale_time=10.0, noise_variance=0.1)
            self.gp_trained = False
            self.prediction_horizon = 5

        self.time = 0.0
        self.step_count = 0

        self.history = {
            'time': [],
            'positions': [],
            'coverage_cost': [],
            'hotspot_distance': [],
        }

    def step(self, dt):
        current_field = self.field.get_field_grid()
        positions = self.swarm.get_positions()

        if self.use_gp:
            self.swarm.sense_all(self.field, self.time)
            if self.step_count % 5 == 0:
                self._update_gp()

            if self.gp_trained:
                predicted_field = self._predict_future_field(dt)
            else:
                predicted_field = current_field
        else:
            predicted_field = current_field

        controls = self.controller.compute_predictive_control(
            positions, current_field, predicted_field,
            alpha=0.6 if self.use_gp else 1.0
        )

        controls = self.cbf.filter_control(positions, controls)
        self.swarm.update_all(controls, dt, use_velocity=True)
        self._record(current_field)

        self.time += dt
        self.step_count += 1

    def _update_gp(self):
        X, y = self.swarm.get_all_sensed_data(time_window=20.0, current_time=self.time)
        if len(X) >= 70:
            try:
                self.gp.fit(X, y)
                self.gp_trained = True
            except:
                pass

    def _predict_future_field(self, dt):
        future_time = self.time + self.prediction_horizon * dt
        x_grid = np.linspace(self. domain[0], self.domain[1], 50)
        y_grid = np.linspace(self.domain[2], self. domain[3], 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        X_pred = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, future_time)])
        try:
            mean, _ = self.gp.predict(X_pred)
            return np.maximum(mean. reshape(50, 50), 0.1)
        except:
            return self.field.get_field_grid()

    def _record(self, field):
        positions = self.swarm.get_positions()
        hotspot_pos = self.field.get_hotspot_position()

        cost = self.voronoi.compute_coverage_cost(positions, field)
        distances = np.linalg.norm(positions - hotspot_pos, axis=1)
        min_dist = np.min(distances)

        self.history['time'].append(self.time)
        self.history['positions'].append(positions. copy())
        self.history['coverage_cost'].append(cost)
        self.history['hotspot_distance'].append(min_dist)

    def get_positions(self):
        return self. swarm.get_positions()


class ComparisonExperiment:
    """GP vs 无GP 对比实验"""

    def __init__(self, num_agents=5, total_time=40.0, dt=0.1, motion_type="linear", seed=42):
        self.num_agents = num_agents
        self. total_time = total_time
        self.dt = dt
        self.motion_type = motion_type
        self.seed = seed
        self.domain = (0, 100, 0, 100)

        self.field = SimpleMovingHotspot(
            domain=self.domain, resolution=50, motion_type=motion_type, seed=seed
        )

        np.random.seed(seed)
        self.sim_with_gp = SingleSimulator(self.field, num_agents, use_gp=True, name="With GP")

        np.random.seed(seed)
        self.sim_without_gp = SingleSimulator(self. field, num_agents, use_gp=False, name="Without GP")

        self. time = 0.0
        self.step_count = 0

        self.field_history = []
        self. hotspot_history = []

        self.output_dir = "output/comparison"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, verbose=True):
        num_steps = int(self.total_time / self.dt)

        if verbose:
            print("Running comparison:  %d steps" % num_steps)
            print("Motion type: %s" % self. motion_type)
            print("UAVs per group: %d" % self.num_agents)
            print("-" * 50)

        for i in range(num_steps):
            self.field.update(self.dt)
            self.sim_with_gp.step(self.dt)
            self.sim_without_gp.step(self.dt)

            if i % 2 == 0:
                self.field_history.append(self.field.get_field_grid().copy())
                self.hotspot_history.append(self.field.get_hotspot_position().copy())

            self.time += self.dt
            self. step_count += 1

            if verbose and i % 50 == 0:
                cost_gp = self.sim_with_gp.history['coverage_cost'][-1]
                cost_no_gp = self.sim_without_gp.history['coverage_cost'][-1]
                dist_gp = self.sim_with_gp.history['hotspot_distance'][-1]
                dist_no_gp = self.sim_without_gp.history['hotspot_distance'][-1]
                print("Step %3d: GP Cost=%.0f, Dist=%.1f | NoGP Cost=%.0f, Dist=%.1f" %
                      (i, cost_gp, dist_gp, cost_no_gp, dist_no_gp))

        if verbose:
            print("-" * 50)
            print("Simulation complete!")

    def print_statistics(self):
        h1 = self.sim_with_gp.history
        h2 = self.sim_without_gp.history

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print("%-25s %15s %15s" % ("Metric", "With GP", "Without GP"))
        print("-" * 60)

        print("%-25s %15.0f %15.0f" % ("Coverage Cost (Avg)", np.mean(h1['coverage_cost']), np.mean(h2['coverage_cost'])))
        print("%-25s %15.0f %15.0f" % ("Coverage Cost (Min)", np.min(h1['coverage_cost']), np.min(h2['coverage_cost'])))
        print("%-25s %15.0f %15.0f" % ("Coverage Cost (Final)", h1['coverage_cost'][-1], h2['coverage_cost'][-1]))

        print("-" * 60)

        print("%-25s %15.1f %15.1f" % ("Hotspot Dist (Avg)", np.mean(h1['hotspot_distance']), np.mean(h2['hotspot_distance'])))
        print("%-25s %15.1f %15.1f" % ("Hotspot Dist (Min)", np.min(h1['hotspot_distance']), np.min(h2['hotspot_distance'])))

        print("-" * 60)

        cost_improve = (np.mean(h2['coverage_cost']) - np.mean(h1['coverage_cost'])) / np.mean(h2['coverage_cost']) * 100
        dist_improve = (np.mean(h2['hotspot_distance']) - np.mean(h1['hotspot_distance'])) / np.mean(h2['hotspot_distance']) * 100

        print("%-25s %14.1f%%" % ("GP Improvement (Cost)", cost_improve))
        print("%-25s %14.1f%%" % ("GP Improvement (Dist)", dist_improve))
        print("=" * 60)

    def plot_comparison_curves(self, save_path=None, show=True):
        """绘制对比曲线"""
        fig, axes = plt. subplots(2, 1, figsize=(12, 8), sharex=True)

        time = np.array(self.sim_with_gp.history['time'])

        ax1 = axes[0]
        cost_gp = np.array(self.sim_with_gp.history['coverage_cost'])
        cost_no_gp = np.array(self.sim_without_gp.history['coverage_cost'])

        ax1.plot(time, cost_gp, 'b-', linewidth=2, label='With GP Prediction')
        ax1.plot(time, cost_no_gp, 'r--', linewidth=2, label='Without GP Prediction')
        ax1.fill_between(time, cost_gp, alpha=0.2, color='blue')
        ax1.fill_between(time, cost_no_gp, alpha=0.2, color='red')

        ax1.set_ylabel('Coverage Cost', fontsize=12)
        ax1.set_title('Coverage Cost Comparison:  GP vs No GP', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        ax2 = axes[1]
        dist_gp = np.array(self.sim_with_gp.history['hotspot_distance'])
        dist_no_gp = np.array(self.sim_without_gp.history['hotspot_distance'])

        ax2.plot(time, dist_gp, 'b-', linewidth=2, label='With GP Prediction')
        ax2.plot(time, dist_no_gp, 'r--', linewidth=2, label='Without GP Prediction')
        ax2.fill_between(time, dist_gp, alpha=0.2, color='blue')
        ax2.fill_between(time, dist_no_gp, alpha=0.2, color='red')

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Min Distance to Hotspot (m)', fontsize=12)
        ax2.set_title('Hotspot Tracking Performance', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "comparison_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved:  %s" % save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_trajectory_comparison(self, frame_idx=-1, save_path=None, show=True):
        """绘制轨迹对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if frame_idx < 0:
            frame_idx = len(self.field_history) + frame_idx
        frame_idx = min(frame_idx, len(self.field_history) - 1)

        data_idx = min(frame_idx * 2, len(self.sim_with_gp.history['positions']) - 1)

        field = self.field_history[frame_idx]
        hotspot = self.hotspot_history[frame_idx]
        current_time = self.sim_with_gp.history['time'][data_idx]

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        sims = [(self.sim_with_gp, "With GP Prediction"), (self.sim_without_gp, "Without GP Prediction")]

        for ax_idx in range(2):
            sim, title = sims[ax_idx]
            ax = axes[ax_idx]

            ax.imshow(field, extent=[0, 100, 0, 100], origin='lower', cmap='YlOrRd', alpha=0.7)
            ax.scatter(hotspot[0], hotspot[1], c='red', s=200, marker='*',
                      edgecolors='darkred', linewidths=2, zorder=20, label='Hotspot')

            circle = plt.Circle(hotspot, self.field.spread, fill=False,
                               color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax.add_patch(circle)

            positions = sim.history['positions'][data_idx]

            for i in range(self.num_agents):
                traj_end = data_idx + 1
                traj_start = max(0, traj_end - 100)
                trajectory = np.array([sim.history['positions'][j][i] for j in range(traj_start, traj_end)])

                if len(trajectory) > 1:
                    alphas = np.linspace(0.1, 0.8, len(trajectory) - 1)
                    for k in range(len(trajectory) - 1):
                        ax.plot(trajectory[k: k+2, 0], trajectory[k:k+2, 1],
                               color=colors[i], alpha=alphas[k], linewidth=2)

                start = sim.history['positions'][0][i]
                ax.scatter(start[0], start[1], c=[colors[i]], s=80,
                          marker='s', alpha=0.5, edgecolors='black', zorder=10)

                ax.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=150,
                          marker='o', edgecolors='black', linewidths=2, zorder=15)

            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel('X (m)', fontsize=11)
            ax.set_ylabel('Y (m)', fontsize=11)
            ax.set_title("%s\nt = %.1fs" % (title, current_time), fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "trajectory_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print("Saved:  %s" % save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_keyframes(self, num_frames=6):
        """绘制关键帧序列"""
        keyframes_dir = os.path.join(self.output_dir, "keyframes")
        os.makedirs(keyframes_dir, exist_ok=True)

        total_snapshots = len(self.field_history)
        indices = np.linspace(0, total_snapshots - 1, num_frames, dtype=int)

        print("Generating %d keyframes..." % num_frames)

        for i, idx in enumerate(indices):
            save_path = os.path.join(keyframes_dir, "frame_%03d.png" % i)
            self.plot_trajectory_comparison(frame_idx=idx, save_path=save_path, show=False)

        print("Keyframes saved to: %s" % keyframes_dir)

    def create_animation(self, save_path=None, fps=15):
        """创建对比动画GIF"""
        print("Creating comparison animation...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = plt.cm.tab10(np. linspace(0, 1, self.num_agents))
        num_frames = len(self.field_history)

        print("Total frames: %d" % num_frames)

        def animate(frame_idx):
            for ax in axes:
                ax.clear()

            data_idx = min(frame_idx * 2, len(self. sim_with_gp.history['positions']) - 1)
            field = self.field_history[frame_idx]
            hotspot = self.hotspot_history[frame_idx]
            current_time = self.sim_with_gp.history['time'][data_idx]

            sims = [(self.sim_with_gp, "With GP"), (self.sim_without_gp, "Without GP")]

            for ax_idx in range(2):
                sim, title = sims[ax_idx]
                ax = axes[ax_idx]

                ax. imshow(field, extent=[0, 100, 0, 100], origin='lower', cmap='YlOrRd', alpha=0.7)
                ax.scatter(hotspot[0], hotspot[1], c='red', s=200, marker='*',
                          edgecolors='darkred', linewidths=2, zorder=20)

                circle = plt. Circle(hotspot, self.field.spread, fill=False,
                                   color='red', linestyle='--', linewidth=2, alpha=0.5)
                ax.add_patch(circle)

                positions = sim.history['positions'][data_idx]

                for i in range(self.num_agents):
                    traj_end = data_idx + 1
                    traj_start = max(0, traj_end - 80)
                    trajectory = np. array([sim.history['positions'][j][i] for j in range(traj_start, traj_end)])

                    if len(trajectory) > 1:
                        ax.plot(trajectory[: , 0], trajectory[:, 1],
                               color=colors[i], alpha=0.5, linewidth=1.5)

                    ax.scatter(positions[i, 0], positions[i, 1], c=[colors[i]], s=120,
                              marker='o', edgecolors='black', linewidths=2, zorder=15)

                dist = sim.history['hotspot_distance'][data_idx]
                ax. set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.set_title("%s | t=%.1fs | Dist=%.1fm" % (title, current_time, dist),
                            fontsize=11, fontweight='bold')
                ax.set_aspect('equal')

            plt.tight_layout()

            if frame_idx % 20 == 0:
                print("  Rendering frame %d/%d" % (frame_idx, num_frames))

            return axes

        anim = FuncAnimation(fig, animate, frames=num_frames, interval=1000 // fps, blit=False)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "comparison_animation.gif")

        print("Saving animation to %s..." % save_path)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
        print("Animation saved:  %s" % save_path)

        plt.close(fig)
        return save_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description='GP vs No-GP Comparison')
    parser.add_argument('--agents', type=int, default=3, help='Number of UAVs per group')
    parser.add_argument('--time', type=float, default=40.0, help='Simulation time (s)')
    parser.add_argument('--motion', type=str, default='circular',
                       choices=['linear', 'circular', 'diagonal'],
                       help='Hotspot motion type')#改成 default='circular'
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--keyframes', type=int, default=6, help='Number of keyframes')
    parser.add_argument('--no-animation', action='store_true', help='Skip animation')
    parser.add_argument('--fps', type=int, default=15, help='Animation FPS')

    args = parser.parse_args()

    print("=" * 60)
    print("GP vs No-GP Comparison Experiment")
    print("=" * 60)
    print("UAVs per group: %d" % args. agents)
    print("Simulation time: %.1fs" % args.time)
    print("Hotspot motion: %s" % args.motion)
    print("Random seed: %d" % args. seed)
    print("=" * 60)

    # 创建实验
    exp = ComparisonExperiment(
        num_agents=args.agents,
        total_time=args.time,
        motion_type=args.motion,
        seed=args.seed
    )

    # 运行实验
    exp.run(verbose=True)

    # 打印统计
    exp.print_statistics()

    # 生成输出
    print("\n" + "=" * 60)
    print("Generating outputs...")
    print("=" * 60)

    # 1. 对比曲线
    print("\n[1/4] Plotting comparison curves...")
    exp.plot_comparison_curves(show=False)

    # 2. 最终轨迹对比
    print("\n[2/4] Plotting final trajectory comparison...")
    exp.plot_trajectory_comparison(frame_idx=-1, show=False)

    # 3. 关键帧
    print("\n[3/4] Generating keyframes...")
    exp.plot_keyframes(num_frames=args.keyframes)

    # 4. 动画
    if not args.no_animation:
        print("\n[4/4] Creating animation (this may take a while)...")
        exp.create_animation(fps=args.fps)
    else:
        print("\n[4/4] Animation skipped (--no-animation flag)")

    print("\n" + "=" * 60)
    print("All outputs saved to: %s" % exp.output_dir)
    print("=" * 60)

    # 最后显示曲线图
    print("\nDisplaying results...")
    exp.plot_comparison_curves(show=True)


if __name__ == "__main__":
    main()