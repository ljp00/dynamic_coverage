"""
多无人机覆盖控制仿真主程序
集成：动态敏感度场 + GP预测 + 覆盖控制 + CBF安全约束 + 任务分配
支持生成动画
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import yaml
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os. path.dirname(os.path. abspath(__file__))))

from src.environment. sensitivity_field import DynamicSensitivityField
from src.prediction.gp_predictor import SpatioTemporalGP, SparseGP
from src.coverage.voronoi import WeightedVoronoi
from src.coverage.lloyd_controller import LloydController
from src.safety.cbf import CBFSafetyFilter, DistributedCBF
from src.safety.mpc_controller import CoverageMPC, MultiAgentMPC
from src.allocation.auction import SequentialAuction, CBBA, Task
from src.agents.uav import UAV, UAVSwarm

# 修复中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class CoverageSimulator:
    """覆盖控制仿真器"""

    def __init__(self, config_path:  str = None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()

        self._parse_config()
        self._init_components()

        self.time = 0.0
        self.step_count = 0
        self. history = {
            'positions':  [],
            'coverage_cost':  [],
            'time': [],
            'field_snapshots': [],
            'cell_snapshots': []
        }

    def _default_config(self) -> dict:
        return {
            'simulation': {
                'dt': 0.1,
                'total_time': 30.0,
                'domain': {'x_min': 0, 'x_max': 100, 'y_min': 0, 'y_max':  100}
            },
            'uav': {
                'num_agents': 5,
                'max_velocity': 5.0,
                'safe_distance': 3.0,
                'sensing_radius': 15.0
            },
            'gp': {
                'length_scale_space': 15.0,
                'length_scale_time': 10.0,
                'noise_variance': 0.1,
                'prediction_horizon': 10
            },
            'coverage': {
                'gain': 1.5,
                'predictive_alpha': 0.6
            },
            'cbf':  {
                'gamma': 1.0,
                'enabled':  True
            },
            'mpc': {
                'enabled':  False,
                'horizon': 15
            },
            'auction': {
                'enabled': False,
                'num_tasks': 5
            }
        }

    def _parse_config(self):
        sim = self.config['simulation']
        self.dt = sim['dt']
        self.total_time = sim['total_time']
        self.domain = (
            sim['domain']['x_min'],
            sim['domain']['x_max'],
            sim['domain']['y_min'],
            sim['domain']['y_max']
        )

        uav = self.config['uav']
        self.num_agents = uav['num_agents']
        self.max_velocity = uav['max_velocity']
        self.safe_distance = uav['safe_distance']
        self. sensing_radius = uav['sensing_radius']

    def _init_components(self):
        # 1. 动态敏感度场
        self.field = DynamicSensitivityField(self.domain, resolution=50)
        self.field.add_random_hotspots(num_hotspots=4, seed=42)

        # 2. 无人机集群
        self.swarm = UAVSwarm(
            self.num_agents,
            self.domain,
            max_velocity=self.max_velocity,
            sensing_radius=self.sensing_radius
        )

        # 3. GP预测器
        gp_config = self.config['gp']
        self.gp = SpatioTemporalGP(
            length_scale_space=gp_config['length_scale_space'],
            length_scale_time=gp_config['length_scale_time'],
            noise_variance=gp_config['noise_variance']
        )
        self.gp_trained = False
        self.prediction_horizon = gp_config['prediction_horizon']

        # 4. 覆盖控制器
        cov_config = self.config['coverage']
        self.coverage_controller = LloydController(
            self.domain,
            gain=cov_config['gain'],
            max_velocity=self.max_velocity,
            resolution=50
        )
        self.predictive_alpha = cov_config['predictive_alpha']

        # 5. CBF安全滤波器
        cbf_config = self.config['cbf']
        self.cbf_enabled = cbf_config['enabled']
        if self.cbf_enabled:
            self.cbf = DistributedCBF(
                safe_distance=self.safe_distance,
                gamma=cbf_config['gamma'],
                domain=self.domain,
                communication_radius=30.0
            )

        # 6. MPC控制器
        mpc_config = self.config['mpc']
        self.mpc_enabled = mpc_config['enabled']
        if self.mpc_enabled:
            self.mpc = MultiAgentMPC(
                num_agents=self.num_agents,
                horizon=mpc_config['horizon'],
                dt=self.dt,
                max_velocity=self.max_velocity,
                safe_distance=self.safe_distance
            )

        # 7. 任务分配
        auction_config = self.config['auction']
        self.auction_enabled = auction_config['enabled']
        if self.auction_enabled:
            self.auction = CBBA(self.num_agents, max_bundle_size=2)
            self.tasks = self._generate_tasks(auction_config. get('num_tasks', 5))
            self.task_allocation = {}

        # Voronoi计算器
        self.voronoi = WeightedVoronoi(self.domain, resolution=50)

    def _generate_tasks(self, num_tasks: int) -> list:
        tasks = []
        for i in range(num_tasks):
            pos = np.array([
                np.random.uniform(self.domain[0] + 10, self.domain[1] - 10),
                np.random.uniform(self.domain[2] + 10, self.domain[3] - 10)
            ])
            priority = np.random.uniform(1.0, 5.0)
            tasks.append(Task(id=i, position=pos, priority=priority))
        return tasks

    def step(self, save_snapshot: bool = True):
        """执行一步仿真"""
        self. field.update(self.dt)
        self.swarm. sense_all(self.field, self.time)

        if self.step_count % 5 == 0:
            self._update_gp()

        current_field = self.field.get_field_grid()

        if self.gp_trained:
            predicted_field = self._predict_future_field()
        else:
            predicted_field = current_field

        positions = self.swarm.get_positions()

        if self.mpc_enabled:
            controls = self._compute_mpc_control(positions, current_field, predicted_field)
        else:
            controls = self. coverage_controller.compute_predictive_control(
                positions,
                current_field,
                predicted_field,
                alpha=self.predictive_alpha
            )

        if self.auction_enabled and self.step_count % 50 == 0:
            self._update_task_allocation()
            controls = self._adjust_for_tasks(controls)

        if self.cbf_enabled:
            controls = self.cbf.filter_control(positions, controls)

        self.swarm.update_all(controls, self.dt, use_velocity=True)
        self._record_history(current_field, save_snapshot)

        self.time += self.dt
        self.step_count += 1

    def _update_gp(self):
        X, y = self.swarm.get_all_sensed_data(time_window=20.0, current_time=self.time)
        if len(X) >= 20:
            try:
                self.gp. fit(X, y)
                self.gp_trained = True
            except:
                pass

    def _predict_future_field(self) -> np.ndarray:
        future_time = self.time + self.prediction_horizon * self.dt
        x_grid = np.linspace(self.domain[0], self.domain[1], 50)
        y_grid = np.linspace(self.domain[2], self.domain[3], 50)
        X, Y = np.meshgrid(x_grid, y_grid)
        X_pred = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, future_time)])
        try:
            mean, var = self.gp.predict(X_pred)
            return mean.reshape(50, 50)
        except:
            return self.field.get_field_grid()

    def _compute_mpc_control(self, positions, current_field, predicted_field):
        cells = self.voronoi.compute_voronoi(positions, predicted_field)
        target_trajectories = np.zeros((self.num_agents, self.mpc.N, 2))
        for i, cell in enumerate(cells):
            target_trajectories[i] = np.tile(cell. centroid, (self.mpc. N, 1))
        velocities = self.swarm.get_velocities()
        controls = self.mpc.solve_centralized(positions, velocities, target_trajectories)
        return velocities + controls * self.dt

    def _update_task_allocation(self):
        positions = self.swarm.get_positions()
        self.task_allocation = self.auction.allocate(positions, self.tasks)

    def _adjust_for_tasks(self, controls):
        positions = self.swarm.get_positions()
        for agent_id, task_ids in self.task_allocation.items():
            if task_ids:
                task = self.tasks[task_ids[0]]
                direction = task.position - positions[agent_id]
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    task_vel = direction / dist * self.max_velocity
                    controls[agent_id] = 0.7 * controls[agent_id] + 0.3 * task_vel
        return controls

    def _record_history(self, field, save_snapshot=True):
        positions = self.swarm.get_positions()
        cost = self.voronoi.compute_coverage_cost(positions, field)
        self.history['positions'].append(positions. copy())
        self.history['coverage_cost'].append(cost)
        self.history['time']. append(self.time)

        if save_snapshot and self.step_count % 2 == 0:
            self. history['field_snapshots'].append(field.copy())
            cells = self.voronoi.compute_voronoi(positions, field)
            self.history['cell_snapshots'].append(cells)

    def run_simulation(self):
        """运行仿真"""
        num_steps = int(self.total_time / self.dt)
        print(f"Running simulation for {num_steps} steps...")
        for i in range(num_steps):
            self.step(save_snapshot=True)
            if i % 50 == 0:
                cost = self.history['coverage_cost'][-1]
                print(f"  Step {i}/{num_steps}, Cost: {cost:.0f}")
        print("Simulation complete!")
        return self.history

    def create_animation(self, save_path: str = 'coverage_animation.gif',
                         fps: int = 15, dpi: int = 100):
        """创建动画"""
        if not self.history['positions']:
            print("No data!  Run simulation first.")
            return

        print("Creating animation...")
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        cost_min = min(self.history['coverage_cost']) * 0.9
        cost_max = max(self.history['coverage_cost']) * 1.1
        num_frames = len(self.history['field_snapshots'])
        print(f"Total frames: {num_frames}")

        def animate(frame_idx):
            for ax in axes:
                ax.clear()

            data_idx = min(frame_idx * 2, len(self.history['positions']) - 1)
            positions = self.history['positions'][data_idx]
            field = self.history['field_snapshots'][frame_idx]
            cells = self.history['cell_snapshots'][frame_idx]
            current_time = self.history['time'][data_idx]

            # 图1
            ax1 = axes[0]
            ax1.imshow(field, extent=[self. domain[0], self.domain[1],
                                      self.domain[2], self. domain[3]],
                       origin='lower', cmap='YlOrRd', alpha=0.7,
                       vmin=0, vmax=np.max(self.history['field_snapshots'][0]) * 1.5)

            for i in range(self.num_agents):
                pos = positions[i]
                cell = cells[i]
                traj_end = data_idx + 1
                traj_start = max(0, traj_end - 100)
                trajectory = np.array([self.history['positions'][j][i]
                                       for j in range(traj_start, traj_end)])
                if len(trajectory) > 1:
                    ax1.plot(trajectory[:, 0], trajectory[:, 1],
                             color=colors[i], alpha=0.4, linewidth=1.5)
                ax1.scatter(pos[0], pos[1], c=[colors[i]], s=120, marker='o',
                            edgecolors='black', linewidths=2, zorder=10)
                ax1.scatter(cell.centroid[0], cell. centroid[1],
                            c=[colors[i]], s=60, marker='x', linewidths=2, zorder=9)
                ax1.plot([pos[0], cell.centroid[0]], [pos[1], cell.centroid[1]],
                         color=colors[i], linestyle='--', alpha=0.6, linewidth=1.5)

            for hotspot in self.field.hotspots:
                ax1.scatter(hotspot. center[0], hotspot.center[1],
                            c='red', s=80, marker='*', zorder=11)

            ax1.set_xlim(self.domain[0], self.domain[1])
            ax1.set_ylim(self.domain[2], self.domain[3])
            ax1.set_title(f'Coverage Control  t = {current_time:.1f}s', fontsize=12, fontweight='bold')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_aspect('equal')

            # 图2
            ax2 = axes[1]
            ax2.imshow(field, extent=[self.domain[0], self.domain[1],
                                      self.domain[2], self.domain[3]],
                       origin='lower', cmap='YlOrRd', alpha=0.4)

            for i in range(self.num_agents):
                pos = positions[i]
                cell = cells[i]
                ax2.scatter(pos[0], pos[1], c=[colors[i]], s=150, marker='o',
                            edgecolors='black', linewidths=2, zorder=10)
                ax2.scatter(cell.centroid[0], cell.centroid[1],
                            c=[colors[i]], s=80, marker='x', linewidths=3, zorder=9)
                ax2.plot([pos[0], cell.centroid[0]], [pos[1], cell. centroid[1]],
                         color=colors[i], linestyle='--', linewidth=2, alpha=0.8)
                ax2.annotate(f'UAV{i}', (pos[0] + 2, pos[1] + 2),
                             fontsize=9, fontweight='bold', color=colors[i])

            ax2.set_xlim(self.domain[0], self.domain[1])
            ax2.set_ylim(self.domain[2], self.domain[3])
            ax2.set_title('Voronoi Partition', fontsize=12, fontweight='bold')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_aspect('equal')

            # 图3
            ax3 = axes[2]
            time_data = self.history['time'][: data_idx + 1]
            cost_data = self.history['coverage_cost'][:data_idx + 1]
            ax3.plot(time_data, cost_data, 'b-', linewidth=2)
            ax3.fill_between(time_data, cost_data, alpha=0.3)

            if len(cost_data) > 0:
                ax3.scatter(time_data[-1], cost_data[-1], c='red', s=100, zorder=10)
                cost_val = cost_data[-1] / 1e6
                ax3.annotate(f'{cost_val:.2f}M',
                             (time_data[-1], cost_data[-1]),
                             textcoords="offset points", xytext=(10, 10),
                             fontsize=10, color='red', fontweight='bold')

            if len(cost_data) > 1:
                min_cost = min(cost_data)
                min_cost_val = min_cost / 1e6
                ax3.axhline(y=min_cost, color='g', linestyle='--', alpha=0.7,
                            label=f'Min: {min_cost_val:.2f}M')
                ax3.legend(loc='upper right')

            ax3.set_xlim(0, self.total_time)
            ax3.set_ylim(cost_min, cost_max)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Coverage Cost')
            ax3.set_title('Coverage Cost Over Time', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

            plt.tight_layout()

            if frame_idx % 20 == 0:
                print(f"  Rendering frame {frame_idx}/{num_frames}")

            return axes

        anim = FuncAnimation(fig, animate, frames=num_frames,
                             interval=1000 // fps, blit=False)

        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=dpi)
        elif save_path.endswith('.mp4'):
            try:
                writer = FFMpegWriter(fps=fps, bitrate=2000)
                anim.save(save_path, writer=writer, dpi=dpi)
            except Exception as e:
                print(f"MP4 failed ({e}), saving as GIF...")
                save_path = save_path.replace('.mp4', '.gif')
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=dpi)
        else:
            writer = PillowWriter(fps=fps)
            anim.save(save_path + '.gif', writer=writer, dpi=dpi)

        print(f"Animation saved to {save_path}")
        plt.close(fig)
        return save_path

    def plot_results(self):
        """绘制最终结果"""
        if not self.history['positions']:
            print("No data to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        # 图1: 轨迹
        ax1 = axes[0, 0]
        field = self.field.get_field_grid()
        ax1.imshow(field, extent=[self.domain[0], self.domain[1],
                                  self.domain[2], self.domain[3]],
                   origin='lower', cmap='YlOrRd', alpha=0.6)

        for i, agent in enumerate(self.swarm. agents):
            traj = np.array(agent.trajectory)
            ax1.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=1.5)
            ax1.scatter(traj[-1, 0], traj[-1, 1], c=[colors[i]], s=120,
                        marker='o', edgecolors='black', linewidths=2, label=f'UAV {i}')
            ax1.scatter(traj[0, 0], traj[0, 1], c=[colors[i]], s=60, marker='s', alpha=0.5)

        ax1.set_title('UAV Trajectories', fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_aspect('equal')

        # 图2: 代价曲线
        ax2 = axes[0, 1]
        ax2.plot(self.history['time'], self.history['coverage_cost'], 'b-', linewidth=2)
        ax2.fill_between(self.history['time'], self.history['coverage_cost'], alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Coverage Cost')
        ax2.set_title('Coverage Cost Over Time', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 图3: 热力图
        ax3 = axes[1, 0]
        all_positions = np.vstack(self.history['positions'])
        h, xedges, yedges = np.histogram2d(
            all_positions[:, 0]. flatten(),
            all_positions[:, 1].flatten(),
            bins=30,
            range=[[self. domain[0], self.domain[1]], [self.domain[2], self.domain[3]]]
        )
        im = ax3.imshow(h. T, origin='lower',
                        extent=[self.domain[0], self.domain[1],
                                self.domain[2], self.domain[3]],
                        cmap='Blues')
        plt.colorbar(im, ax=ax3, label='Visit frequency')
        ax3.set_title('Position Heatmap', fontweight='bold')
        ax3.set_aspect('equal')

        # 图4: 统计
        ax4 = axes[1, 1]
        costs = self.history['coverage_cost']
        metrics = ['Initial', 'Final', 'Min', 'Average']
        values = [costs[0], costs[-1], min(costs), np.mean(costs)]
        bar_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        bars = ax4.bar(metrics, values, color=bar_colors)

        for bar, val in zip(bars, values):
            val_m = val / 1e6
            ax4.text(bar. get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f'{val_m:.2f}M', ha='center', va='bottom', fontsize=10)

        improvement = (costs[0] - costs[-1]) / costs[0] * 100
        ax4.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
                 transform=ax4.transAxes, ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round',
                           facecolor='lightgreen' if improvement > 0 else 'lightcoral',
                           alpha=0.5))
        ax4.set_ylabel('Coverage Cost')
        ax4.set_title('Performance Statistics', fontweight='bold')

        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
        print("Results saved to simulation_results.png")
        plt.show()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-UAV Coverage Control Simulation')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--agents', type=int, default=5, help='Number of UAVs')
    parser.add_argument('--time', type=float, default=30.0, help='Simulation duration (s)')
    parser.add_argument('--save-gif', type=str, default='coverage_animation.gif', help='Output GIF path')
    parser.add_argument('--fps', type=int, default=15, help='Animation FPS')
    parser.add_argument('--no-show', action='store_true', help='Do not show result window')
    parser.add_argument('--cbf', action='store_true', default=True, help='Enable CBF')

    args = parser.parse_args()

    # 创建仿真器
    sim = CoverageSimulator(config_path=args.config)

    # 更新配置
    sim.config['uav']['num_agents'] = args.agents
    sim.config['simulation']['total_time'] = args. time
    sim.config['cbf']['enabled'] = args.cbf
    sim._parse_config()
    sim._init_components()

    print("=" * 60)
    print("Multi-UAV Coverage Control Simulation")
    print("=" * 60)
    print(f"Number of UAVs: {sim. num_agents}")
    print(f"Simulation time: {sim. total_time}s")
    print(f"CBF Safety:  {'Enabled' if sim.cbf_enabled else 'Disabled'}")
    print(f"Output:  {args.save_gif}")
    print("=" * 60)

    # 1. 运行仿真
    history = sim.run_simulation()

    # 2. 打印统计
    print("\n" + "=" * 60)
    print("Simulation Statistics:")
    print("=" * 60)
    initial = history['coverage_cost'][0]
    final = history['coverage_cost'][-1]
    minimum = min(history['coverage_cost'])
    improvement = (initial - final) / initial * 100
    print(f"  Initial cost: {initial:.0f}")
    print(f"  Final cost:    {final:.0f}")
    print(f"  Min cost:     {minimum:.0f}")
    print(f"  Improvement:  {improvement:.1f}%")

    # 3. 生成动画
    sim.create_animation(save_path=args.save_gif, fps=args.fps, dpi=100)

    # 4. 绘制结果
    if not args.no_show:
        sim.plot_results()


if __name__ == "__main__":
    main()