"""
多无人机覆盖控制仿真主程序
集成：动态敏感度场 + GP预测 + 覆盖控制 + CBF安全约束 + 任务分配
"""
import numpy as np
import matplotlib. pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            'positions': [],
            'coverage_cost': [],
            'time': []
        }

    def _default_config(self) -> dict:
        return {
            'simulation': {
                'dt': 0.1,
                'total_time': 60.0,
                'domain': {'x_min': 0, 'x_max':  100, 'y_min': 0, 'y_max':  100}
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
                'enabled': True
            },
            'mpc': {
                'enabled': False,
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
        mpc_config = self. config['mpc']
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

    def step(self):
        """执行一步仿真"""
        # 1. 更新敏感度场
        self. field.update(self.dt)

        # 2. 所有无人机感知
        self.swarm. sense_all(self.field, self.time)

        # 3. 更新GP模型
        if self.step_count % 5 == 0:
            self._update_gp()

        # 4. 获取当前和预测的敏感度场
        current_field = self.field.get_field_grid()

        if self.gp_trained:
            predicted_field = self._predict_future_field()
        else:
            predicted_field = current_field

        # 5. 计算覆盖控制
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

        # 6. 任务分配调整
        if self.auction_enabled and self.step_count % 50 == 0:
            self._update_task_allocation()
            controls = self._adjust_for_tasks(controls)

        # 7. CBF安全滤波
        if self.cbf_enabled:
            controls = self.cbf.filter_control(positions, controls)

        # 8. 更新无人机状态
        self.swarm. update_all(controls, self.dt, use_velocity=True)

        # 9. 记录历史
        self._record_history(current_field)

        # 10. 更新时间
        self.time += self.dt
        self.step_count += 1

    def _update_gp(self):
        X, y = self.swarm.get_all_sensed_data(time_window=20.0, current_time=self.time)
        if len(X) >= 20:
            try:
                self.gp. fit(X, y)
                self.gp_trained = True
            except Exception as e:
                print(f"GP training failed: {e}")

    def _predict_future_field(self) -> np.ndarray:
        future_time = self.time + self.prediction_horizon * self.dt

        x_grid = np.linspace(self.domain[0], self.domain[1], 50)
        y_grid = np.linspace(self.domain[2], self.domain[3], 50)
        X, Y = np.meshgrid(x_grid, y_grid)

        X_pred = np.column_stack([
            X.ravel(),
            Y.ravel(),
            np.full(X.size, future_time)
        ])

        try:
            mean, var = self.gp.predict(X_pred)
            return mean.reshape(50, 50)
        except:
            return self.field.get_field_grid()

    def _compute_mpc_control(self, positions, current_field, predicted_field):
        voronoi = WeightedVoronoi(self.domain, resolution=50)
        cells = voronoi.compute_voronoi(positions, predicted_field)

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

    def _record_history(self, field):
        positions = self.swarm.get_positions()
        cost = self.coverage_controller.voronoi. compute_coverage_cost(positions, field)

        self.history['positions'].append(positions. copy())
        self.history['coverage_cost'].append(cost)
        self.history['time']. append(self.time)

    def run(self, visualize: bool = True, save_animation: bool = False):
        """运行完整仿真"""
        num_steps = int(self.total_time / self.dt)

        if visualize:
            self._run_with_visualization(num_steps, save_animation)
        else:
            self._run_headless(num_steps)

        return self.history

    def _run_headless(self, num_steps:  int):
        """无可视化运行"""
        print("Starting simulation...")
        for i in range(num_steps):
            self.step()
            if i % 100 == 0:
                print(f"Progress: {i}/{num_steps}, Coverage cost: {self.history['coverage_cost'][-1]:.2f}")
        print("Simulation complete!")

    def _run_with_visualization(self, num_steps: int, save_animation: bool):
        """带实时可视化运行"""
        # 先执行所有仿真步骤
        print("Running simulation...")
        for i in range(num_steps):
            self.step()
            if i % 100 == 0:
                print(f"Progress: {i}/{num_steps}, Coverage cost: {self.history['coverage_cost'][-1]:.2f}")
        print("Simulation complete!  Generating visualization...")

        # 然后创建动画
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))

        # 获取最终状态用于可视化
        positions = self.swarm.get_positions()
        field = self.field.get_field_grid()

        # 图1: 最终状态
        ax1 = axes[0]
        ax1.imshow(
            field,
            extent=[self.domain[0], self.domain[1], self.domain[2], self.domain[3]],
            origin='lower',
            cmap='YlOrRd',
            alpha=0.7
        )

        voronoi = WeightedVoronoi(self.domain, resolution=50)
        cells = voronoi.compute_voronoi(positions, field)

        for i, agent in enumerate(self.swarm. agents):
            pos = positions[i]
            # 绘制轨迹
            traj = np.array(agent.trajectory)
            ax1.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.5, linewidth=1)
            # 绘制最终位置
            ax1.scatter(pos[0], pos[1], c=[colors[i]], s=150, marker='o',
                        edgecolors='black', linewidths=2, zorder=5, label=f'UAV {i}')
            # 绘制质心
            ax1.scatter(cells[i].centroid[0], cells[i].centroid[1],
                        c=[colors[i]], s=50, marker='x', zorder=4)
            ax1.plot([pos[0], cells[i].centroid[0]],
                     [pos[1], cells[i].centroid[1]],
                     color=colors[i], linestyle='--', alpha=0.5)
            # 绘制起点
            ax1.scatter(traj[0, 0], traj[0, 1], c=[colors[i]], s=80, marker='s', alpha=0.5)

        for hotspot in self.field.hotspots:
            ax1.scatter(hotspot. center[0], hotspot.center[1],
                        c='red', s=100, marker='*', zorder=6)

        ax1.set_xlim(self.domain[0], self.domain[1])
        ax1.set_ylim(self.domain[2], self. domain[3])
        ax1.set_title(f'UAV Trajectories (Final state at t={self.time:.1f}s)')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_aspect('equal')

        # 图2: Voronoi分割
        ax2 = axes[1]
        ax2.imshow(
            field,
            extent=[self.domain[0], self.domain[1], self. domain[2], self.domain[3]],
            origin='lower',
            cmap='YlOrRd',
            alpha=0.5
        )

        for i, cell in enumerate(cells):
            ax2.scatter(positions[i, 0], positions[i, 1],
                        c=[colors[i]], s=150, marker='o', edgecolors='black',
                        linewidths=2, zorder=10)
            ax2.scatter(cell.centroid[0], cell. centroid[1],
                        c=[colors[i]], s=80, marker='x', linewidths=2, zorder=9)
            ax2.plot([positions[i, 0], cell.centroid[0]],
                     [positions[i, 1], cell.centroid[1]],
                     color=colors[i], linestyle='--', alpha=0.7)
            ax2.annotate(f'UAV{i}', (positions[i, 0] + 2, positions[i, 1] + 2))

        ax2.set_xlim(self.domain[0], self.domain[1])
        ax2.set_ylim(self.domain[2], self.domain[3])
        ax2.set_title('Voronoi Partition & Centroids')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_aspect('equal')

        # 图3: 覆盖代价曲线
        ax3 = axes[2]
        ax3.plot(self.history['time'], self.history['coverage_cost'], 'b-', linewidth=2)
        ax3.fill_between(self.history['time'], self.history['coverage_cost'], alpha=0.2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Coverage Cost')
        ax3.set_title('Coverage Cost Over Time')
        ax3.grid(True, alpha=0.3)

        min_cost = min(self.history['coverage_cost'])
        ax3.axhline(y=min_cost, color='g', linestyle='--', alpha=0.5,
                    label=f'Min:  {min_cost:.1f}')
        ax3.legend()

        plt.tight_layout()

        if save_animation:
            plt.savefig('coverage_simulation.png', dpi=150, bbox_inches='tight')
            print("Figure saved to coverage_simulation.png")

        plt.show(block=True)

    def plot_results(self):
        """绘制仿真结果"""
        # 检查是否有数据
        if not self.history['positions']:
            print("No simulation data to plot.  Please run simulation first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax1 = axes[0, 0]
        field = self.field.get_field_grid()
        ax1.imshow(
            field,
            extent=[self.domain[0], self.domain[1], self.domain[2], self.domain[3]],
            origin='lower',
            cmap='YlOrRd',
            alpha=0.6
        )

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_agents))
        for i, agent in enumerate(self.swarm.agents):
            traj = np.array(agent. trajectory)
            ax1.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7, linewidth=1)
            ax1.scatter(traj[-1, 0], traj[-1, 1], c=[colors[i]], s=100,
                        marker='o', edgecolors='black', linewidths=2, label=f'UAV {i}')
            ax1.scatter(traj[0, 0], traj[0, 1], c=[colors[i]], s=50, marker='s', alpha=0.5)

        ax1.set_title('UAV Trajectories')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_aspect('equal')

        ax2 = axes[0, 1]
        ax2.plot(self.history['time'], self.history['coverage_cost'], 'b-', linewidth=2)
        ax2.fill_between(self.history['time'], self.history['coverage_cost'], alpha=0.2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Coverage Cost')
        ax2.set_title('Coverage Cost Over Time')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        all_positions = np.vstack(self.history['positions'])
        h, xedges, yedges = np.histogram2d(
            all_positions[:, 0]. flatten(),
            all_positions[:, 1].flatten(),
            bins=30,
            range=[[self. domain[0], self.domain[1]], [self.domain[2], self.domain[3]]]
        )
        im = ax3.imshow(h. T, origin='lower', extent=[self.domain[0], self.domain[1],
                                                     self.domain[2], self. domain[3]],
                        cmap='Blues')
        plt.colorbar(im, ax=ax3, label='Visit frequency')
        ax3.set_title('UAV Position Heatmap')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_aspect('equal')

        ax4 = axes[1, 1]
        costs = self.history['coverage_cost']
        initial_cost = costs[0]
        final_cost = costs[-1]
        min_cost = min(costs)
        avg_cost = np.mean(costs)

        metrics = ['Initial', 'Final', 'Min', 'Average']
        values = [initial_cost, final_cost, min_cost, avg_cost]
        bar_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        bars = ax4.bar(metrics, values, color=bar_colors)
        ax4.set_ylabel('Coverage Cost')
        ax4.set_title('Performance Statistics')
        ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9)

        improvement = (initial_cost - final_cost) / initial_cost * 100
        ax4.text(0.5, 0.95, f'Improvement: {improvement:.1f}%',
                 transform=ax4.transAxes, ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round',
                           facecolor='lightgreen' if improvement > 0 else 'lightcoral',
                           alpha=0.5))

        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
        print("Results saved to simulation_results.png")
        plt.show(block=True)


# ================================================================
# 主函数 - 必须在类的外部
# ================================================================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-UAV Coverage Control Simulation')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--save-anim', action='store_true', help='Save animation')
    parser.add_argument('--agents', type=int, default=5, help='Number of UAVs')
    parser.add_argument('--time', type=float, default=60.0, help='Simulation duration')
    parser.add_argument('--cbf', action='store_true', default=True, help='Enable CBF')
    parser.add_argument('--mpc', action='store_true', help='Enable MPC')
    parser.add_argument('--auction', action='store_true', help='Enable task allocation')

    args = parser.parse_args()

    # 创建仿真器
    sim = CoverageSimulator(config_path=args.config)

    # 覆盖命令行参数
    if args.config is None:
        sim.config['uav']['num_agents'] = args.agents
        sim.config['simulation']['total_time'] = args.time
        sim.config['cbf']['enabled'] = args.cbf
        sim.config['mpc']['enabled'] = args.mpc
        sim.config['auction']['enabled'] = args.auction
        sim._parse_config()
        sim._init_components()

    print("=" * 60)
    print("Multi-UAV Dynamic Coverage Control Simulation")
    print("=" * 60)
    print(f"Number of UAVs: {sim. num_agents}")
    print(f"Simulation time: {sim.total_time}s")
    print(f"CBF Safety:  {'Enabled' if sim.cbf_enabled else 'Disabled'}")
    print(f"MPC Control: {'Enabled' if sim. mpc_enabled else 'Disabled'}")
    print(f"Task Allocation: {'Enabled' if sim.auction_enabled else 'Disabled'}")
    print("=" * 60)

    # 运行仿真
    history = sim.run(visualize=not args.no_viz, save_animation=args.save_anim)

    # 绘制结果（仅在有数据时）
    if history['coverage_cost']:
        sim.plot_results()

        print("\n" + "=" * 60)
        print("Simulation Statistics:")
        print("=" * 60)
        print(f"  Initial cost: {history['coverage_cost'][0]:.2f}")
        print(f"  Final cost:    {history['coverage_cost'][-1]:.2f}")
        print(f"  Min cost:     {min(history['coverage_cost']):.2f}")
        improvement = (history['coverage_cost'][0] - history['coverage_cost'][-1]) / history['coverage_cost'][0] * 100
        print(f"  Improvement:   {improvement:.1f}%")
    else:
        print("No simulation data collected.")


if __name__ == "__main__":
    main()