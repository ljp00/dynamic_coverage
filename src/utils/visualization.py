"""
可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple, Dict
import matplotlib.colors as mcolors


class CoverageVisualizer:
    """覆盖控制可视化器"""

    def __init__(self, domain: Tuple[float, float, float, float],
                 num_agents: int,
                 figsize: Tuple[int, int] = (16, 6)):
        """
        Args:
            domain: 工作区域 (x_min, x_max, y_min, y_max)
            num_agents: 无人机数量
            figsize: 图像大小
        """
        self.domain = domain
        self.num_agents = num_agents
        self. figsize = figsize

        # 颜色映射
        self. agent_colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

        # 图形对象
        self.fig = None
        self.axes = None
        self.initialized = False

    def setup_figure(self, num_subplots: int = 3):
        """初始化图形"""
        self.fig, self.axes = plt.subplots(1, num_subplots, figsize=self.figsize)
        if num_subplots == 1:
            self.axes = [self.axes]
        self.initialized = True
        return self.fig, self.axes

    def plot_sensitivity_field(self, ax: plt.Axes,
                               field:  np.ndarray,
                               title: str = "敏感度场",
                               cmap: str = 'YlOrRd',
                               alpha: float = 0.7,
                               show_colorbar: bool = True):
        """
        绘制敏感度场

        Args:
            ax: matplotlib轴对象
            field: 敏感度场网格 shape (H, W)
            title: 标题
            cmap: 颜色映射
            alpha: 透明度
            show_colorbar: 是否显示颜色条
        """
        im = ax.imshow(
            field,
            extent=[self.domain[0], self. domain[1],
                   self.domain[2], self. domain[3]],
            origin='lower',
            cmap=cmap,
            alpha=alpha,
            aspect='equal'
        )

        if show_colorbar:
            plt.colorbar(im, ax=ax, shrink=0.8, label='敏感度')

        ax.set_xlim(self.domain[0], self.domain[1])
        ax.set_ylim(self.domain[2], self.domain[3])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)

        return im

    def plot_agents(self, ax: plt. Axes,
                    positions: np.ndarray,
                    velocities: np.ndarray = None,
                    size: int = 100,
                    show_labels: bool = True,
                    show_velocity: bool = True):
        """
        绘制无人机位置

        Args:
            ax: matplotlib轴对象
            positions: 无人机位置 shape (N, 2)
            velocities: 无人机速度 shape (N, 2)
            size: 标记大小
            show_labels: 是否显示标签
            show_velocity: 是否显示速度箭头
        """
        for i, pos in enumerate(positions):
            # 无人机标记
            ax.scatter(pos[0], pos[1],
                      c=[self.agent_colors[i]],
                      s=size,
                      marker='o',
                      edgecolors='black',
                      linewidths=2,
                      zorder=10)

            # 标签
            if show_labels:
                ax.annotate(f'{i}',
                           (pos[0] + 1.5, pos[1] + 1.5),
                           fontsize=9,
                           fontweight='bold',
                           zorder=11)

            # 速度箭头
            if show_velocity and velocities is not None:
                vel = velocities[i]
                speed = np.linalg.norm(vel)
                if speed > 0.1:
                    ax.arrow(pos[0], pos[1],
                            vel[0] * 0.8, vel[1] * 0.8,
                            head_width=1.0,
                            head_length=0.5,
                            fc=self.agent_colors[i],
                            ec='black',
                            alpha=0.7,
                            zorder=9)

    def plot_trajectories(self, ax: plt. Axes,
                         trajectories: List[np.ndarray],
                         max_points: int = 100,
                         alpha: float = 0.5,
                         linewidth: float = 1.5):
        """
        绘制无人机轨迹

        Args:
            ax: matplotlib轴对象
            trajectories: 轨迹列表，每个元素 shape (T, 2)
            max_points: 最多显示的轨迹点数
            alpha: 透明度
            linewidth: 线宽
        """
        for i, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue

            # 截取最近的轨迹点
            traj = np.array(traj[-max_points:])

            # 渐变颜色效果
            points = traj. reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # 透明度渐变
            alphas = np.linspace(0.1, alpha, len(segments))
            colors = np.zeros((len(segments), 4))
            colors[:, : 3] = self.agent_colors[i][: 3]
            colors[:, 3] = alphas

            lc = LineCollection(segments, colors=colors, linewidths=linewidth)
            ax.add_collection(lc)

    def plot_voronoi_cells(self, ax: plt.Axes,
                          positions: np.ndarray,
                          cells: List,
                          show_centroids: bool = True,
                          show_connections: bool = True,
                          alpha: float = 0.2):
        """
        绘制Voronoi分割

        Args:
            ax: matplotlib轴对象
            positions: 无人机位置
            cells: Voronoi单元列表
            show_centroids:  是否显示质心
            show_connections: 是否显示到质心的连线
            alpha: 填充透明度
        """
        for i, cell in enumerate(cells):
            # 质心
            if show_centroids:
                ax.scatter(cell. centroid[0], cell.centroid[1],
                          c=[self.agent_colors[i]],
                          s=60,
                          marker='x',
                          linewidths=2,
                          zorder=8)

            # 连接线
            if show_connections:
                ax.plot([positions[i][0], cell.centroid[0]],
                       [positions[i][1], cell.centroid[1]],
                       color=self.agent_colors[i],
                       linestyle='--',
                       alpha=0.6,
                       linewidth=1.5,
                       zorder=7)

    def plot_hotspots(self, ax: plt.Axes,
                     hotspots: List,
                     marker:  str = '*',
                     size: int = 150,
                     color: str = 'red'):
        """绘制热点位置"""
        for hotspot in hotspots:
            ax.scatter(hotspot. center[0], hotspot.center[1],
                      c=color, s=size, marker=marker,
                      edgecolors='darkred', linewidths=1,
                      zorder=12, alpha=0.8)
            # 绘制扩散范围
            circle = Circle(hotspot.center, hotspot. spread,
                           fill=False, color='red',
                           linestyle=':', alpha=0.3)
            ax.add_patch(circle)

    def plot_tasks(self, ax: plt. Axes,
                   tasks: List,
                   allocation: Dict[int, List[int]] = None):
        """
        绘制任务点

        Args:
            ax: matplotlib轴对象
            tasks: 任务列表
            allocation: 任务分配 {agent_id: [task_ids]}
        """
        for task in tasks:
            # 确定颜色（根据分配情况）
            color = 'gray'
            if allocation:
                for agent_id, task_ids in allocation.items():
                    if task. id in task_ids:
                        color = self.agent_colors[agent_id]
                        break

            ax.scatter(task.position[0], task.position[1],
                      c=[color] if isinstance(color, np.ndarray) else color,
                      s=100, marker='s',
                      edgecolors='black', linewidths=1.5,zorder=11)
            ax.annotate(f'T{task.id}',
                       (task.position[0] + 2, task.position[1] + 2),
                       fontsize=8, fontweight='bold')

    def plot_safety_regions(self, ax: plt. Axes,
                           positions: np.ndarray,
                           safe_distance: float,
                           alpha: float = 0.1):
        """绘制安全区域"""
        for i, pos in enumerate(positions):
            circle = Circle(pos, safe_distance / 2,
                           fill=True,
                           facecolor=self.agent_colors[i],
                           edgecolor=self.agent_colors[i],
                           alpha=alpha,
                           linestyle='-',
                           linewidth=1)
            ax.add_patch(circle)

    def plot_communication_graph(self, ax: plt.Axes,
                                 positions: np.ndarray,
                                 comm_radius: float,
                                 alpha: float = 0.3):
        """绘制通信拓扑"""
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= comm_radius:
                    ax. plot([positions[i][0], positions[j][0]],
                           [positions[i][1], positions[j][1]],
                           'g-', alpha=alpha, linewidth=1, zorder=1)

    def plot_cost_curve(self, ax: plt. Axes,
                        times: List[float],
                        costs:  List[float],
                        title: str = "覆盖代价"):
        """绘制代价曲线"""
        ax.clear()
        ax.plot(times, costs, 'b-', linewidth=2)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('覆盖代价')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if len(costs) > 0:
            min_cost = min(costs)
            ax.axhline(y=min_cost, color='g', linestyle='--',
                      alpha=0.5, label=f'最小:  {min_cost:.1f}')
            ax.legend(loc='upper right')

    def plot_gp_uncertainty(self, ax: plt. Axes,
                           variance: np.ndarray,
                           title: str = "GP预测不确定性"):
        """绘制GP预测不确定性"""
        im = ax.imshow(
            np.sqrt(variance),  # 标准差
            extent=[self. domain[0], self.domain[1],
                   self.domain[2], self.domain[3]],
            origin='lower',
            cmap='Blues',
            alpha=0.7,
            aspect='equal'
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label='标准差')
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        return im


class AnimationRecorder:
    """动画录制器"""

    def __init__(self, visualizer: CoverageVisualizer):
        self.visualizer = visualizer
        self.frames = []

    def capture_frame(self, **data):
        """捕获一帧数据"""
        self.frames.append(data. copy())

    def create_animation(self, interval: int = 50) -> FuncAnimation:
        """创建动画"""
        if not self.frames:
            raise ValueError("没有捕获的帧数据")

        fig, axes = self.visualizer.setup_figure(num_subplots=3)

        def update(frame_idx):
            data = self.frames[frame_idx]

            for ax in axes:
                ax. clear()

            # 绘制各个组件...
            # (具体实现取决于数据结构)

            return axes

        anim = FuncAnimation(
            fig, update, frames=len(self.frames),
            interval=interval, blit=False
        )

        return anim

    def save_animation(self, filename: str, fps: int = 20):
        """保存动画"""
        anim = self.create_animation(interval=1000 // fps)
        anim.save(filename, writer='ffmpeg', fps=fps)
        print(f"动画已保存至: {filename}")


def plot_simulation_summary(history: Dict,
                           domain: Tuple[float, float, float, float],
                           trajectories: List[np.ndarray],
                           save_path: str = None):
    """
    绘制仿真总结图

    Args:
        history: 仿真历史数据
        domain: 工作区域
        trajectories: 所有无人机轨迹
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    num_agents = len(trajectories)
    colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

    # 图1: 轨迹总览
    ax1 = axes[0, 0]
    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        ax1.plot(traj[:, 0], traj[:, 1],
                color=colors[i], alpha=0.7, linewidth=1.5,
                label=f'UAV {i}')
        ax1.scatter(traj[0, 0], traj[0, 1],
                   c=[colors[i]], s=80, marker='s', alpha=0.5)
        ax1.scatter(traj[-1, 0], traj[-1, 1],
                   c=[colors[i]], s=100, marker='o',
                   edgecolors='black', linewidths=2)
    ax1.set_xlim(domain[0], domain[1])
    ax1.set_ylim(domain[2], domain[3])
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('无人机轨迹')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 图2: 覆盖代价
    ax2 = axes[0, 1]
    times = history. get('time', [])
    costs = history.get('coverage_cost', [])
    if times and costs:
        ax2.plot(times, costs, 'b-', linewidth=2)
        ax2.fill_between(times, costs, alpha=0.2)
        ax2.axhline(y=min(costs), color='g', linestyle='--',
                   alpha=0.7, label=f'最小:  {min(costs):.1f}')
        ax2.axhline(y=np.mean(costs), color='orange', linestyle=':',
                   alpha=0.7, label=f'平均: {np.mean(costs):.1f}')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('覆盖代价')
    ax2.set_title('覆盖代价变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: 位置热力图
    ax3 = axes[1, 0]
    all_positions = []
    for traj in trajectories:
        all_positions.extend(traj)
    all_positions = np.array(all_positions)

    h, xedges, yedges = np.histogram2d(
        all_positions[:, 0],
        all_positions[:, 1],
        bins=40,
        range=[[domain[0], domain[1]], [domain[2], domain[3]]]
    )
    im = ax3.imshow(h.T, origin='lower',
                    extent=[domain[0], domain[1], domain[2], domain[3]],
                    cmap='hot', aspect='equal')
    plt.colorbar(im, ax=ax3, shrink=0.8, label='访问频次')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('位置分布热力图')

    # 图4: 性能统计
    ax4 = axes[1, 1]
    if costs:
        metrics = {
            '初始代价': costs[0],
            '最终代价': costs[-1],
            '最小代价': min(costs),
            '平均代价':  np.mean(costs),
            '代价标准差': np.std(costs)
        }

        bars = ax4.bar(metrics. keys(), metrics.values(),
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'])
        ax4.set_ylabel('代价值')
        ax4.set_title('性能统计')
        ax4.tick_params(axis='x', rotation=15)

        for bar, val in zip(bars, metrics.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

        # 添加改善率
        if costs[0] > 0:
            improvement = (costs[0] - costs[-1]) / costs[0] * 100
            ax4.text(0.5, 0.95, f'总体改善率: {improvement:.1f}%',
                    transform=ax4.transAxes, ha='center',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"结果图已保存至: {save_path}")

    plt.show()


def plot_comparison(results: Dict[str, Dict],
                   title: str = "算法对比"):
    """
    绘制多种算法的对比图

    Args:
        results: {算法名: {'time': [... ], 'cost': [...]}}
        title: 图标题
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # 代价曲线对比
    ax1 = axes[0]
    for i, (name, data) in enumerate(results. items()):
        ax1.plot(data['time'], data['cost'],
                color=colors[i], linewidth=2, label=name)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('覆盖代价')
    ax1.set_title('覆盖代价对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 最终性能柱状图
    ax2 = axes[1]
    names = list(results.keys())
    final_costs = [results[n]['cost'][-1] for n in names]
    min_costs = [min(results[n]['cost']) for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax2.bar(x - width/2, final_costs, width, label='最终代价', color='steelblue')
    bars2 = ax2.bar(x + width/2, min_costs, width, label='最小代价', color='lightcoral')

    ax2.set_ylabel('代价值')
    ax2.set_title('性能对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()

    plt.tight_layout()
    plt.show()