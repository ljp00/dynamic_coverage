"""
简化测试脚本 - 测试仿真框架是否正常工作
"""
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Windows 推荐使用 TkAgg
import matplotlib.pyplot as plt

print("=" * 60)
print("开始测试多无人机覆盖控制仿真框架")
print("=" * 60)

# 测试1:  导入模块
print("\n[1/6] 测试模块导入...")
try:
    from src.environment.sensitivity_field import DynamicSensitivityField
    from src.prediction.gp_predictor import SpatioTemporalGP
    from src.coverage.voronoi import WeightedVoronoi
    from src.coverage.lloyd_controller import LloydController
    from src.safety.cbf import DistributedCBF
    from src.agents.uav import UAVSwarm

    print("  ✓ 所有模块导入成功!")
except ImportError as e:
    print(f"  ✗ 模块导入失败: {e}")
    sys.exit(1)

# 测试2: 创建敏感度场
print("\n[2/6] 测试敏感度场...")
domain = (0, 100, 0, 100)
field = DynamicSensitivityField(domain, resolution=50)
field.add_random_hotspots(num_hotspots=3, seed=42)
print(f"  ✓ 敏感度场创建成功!  热点数量: {len(field.hotspots)}")

# 测试3: 创建无人机集群
print("\n[3/6] 测试无人机集群...")
num_agents = 5
swarm = UAVSwarm(num_agents, domain, max_velocity=5.0)
positions = swarm.get_positions()
print(f"  ✓ 无人机集群创建成功! 数量: {num_agents}")
print(f"  初始位置:\n{positions}")

# 测试4: 测试覆盖控制器
print("\n[4/6] 测试覆盖控制器...")
controller = LloydController(domain, gain=1.5, max_velocity=5.0)
field_grid = field.get_field_grid()
velocities, cells = controller.compute_control(positions, field_grid)
print(f"  ✓ 覆盖控制器正常!  计算得到速度:\n{velocities}")

# 测试5: 测试CBF安全滤波
print("\n[5/6] 测试CBF安全滤波...")
cbf = DistributedCBF(safe_distance=3.0, gamma=1.0, domain=domain)
safe_velocities = cbf.filter_control(positions, velocities)
print(f"  ✓ CBF滤波正常! 安全速度:\n{safe_velocities}")

# 测试6: 运行简单仿真并可视化
print("\n[6/6] 运行简单仿真...")

# 仿真参数
dt = 0.1
num_steps = 200
history_positions = [positions.copy()]
history_cost = []

# Voronoi计算器
voronoi = WeightedVoronoi(domain)

for step in range(num_steps):
    # 更新敏感度场
    field.update(dt)
    field_grid = field.get_field_grid()

    # 获取当前位置
    positions = swarm.get_positions()

    # 计算覆盖控制
    velocities, cells = controller.compute_control(positions, field_grid)

    # CBF安全滤波
    safe_velocities = cbf.filter_control(positions, velocities)

    # 更新无人机
    swarm.update_all(safe_velocities, dt)

    # 记录数据
    history_positions.append(swarm.get_positions().copy())
    cost = voronoi.compute_coverage_cost(positions, field_grid)
    history_cost.append(cost)

    if step % 50 == 0:
        print(f"  步数:  {step}/{num_steps}, 覆盖代价: {cost:.2f}")

print(f"\n  ✓ 仿真完成!")
print(f"  初始代价: {history_cost[0]:.2f}")
print(f"  最终代价:  {history_cost[-1]:.2f}")
print(f"  改善率: {(history_cost[0] - history_cost[-1]) / history_cost[0] * 100:.1f}%")

# 绘制结果
print("\n正在生成可视化图形...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
colors = plt.cm.tab10(np.linspace(0, 1, num_agents))

# 图1: 最终状态
ax1 = axes[0, 0]
field_grid = field.get_field_grid()
im = ax1.imshow(field_grid, extent=[0, 100, 0, 100], origin='lower',
                cmap='YlOrRd', alpha=0.7)
plt.colorbar(im, ax=ax1, label='敏感度')

final_positions = swarm.get_positions()
for i, agent in enumerate(swarm.agents):
    # 绘制轨迹
    traj = np.array(agent.trajectory)
    ax1.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.5, linewidth=1)
    # 绘制最终位置
    ax1.scatter(final_positions[i, 0], final_positions[i, 1],
                c=[colors[i]], s=150, marker='o', edgecolors='black',
                linewidths=2, zorder=10, label=f'UAV {i}')
    # 绘制起始位置
    ax1.scatter(traj[0, 0], traj[0, 1], c=[colors[i]], s=80,
                marker='s', alpha=0.5, zorder=9)

# 绘制热点
for hotspot in field.hotspots:
    ax1.scatter(hotspot.center[0], hotspot.center[1], c='red', s=100,
                marker='*', zorder=11)

ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('无人机轨迹与敏感度场')
ax1.legend(loc='upper right', fontsize=8)
ax1.set_aspect('equal')

# 图2:  Voronoi分割
ax2 = axes[0, 1]
ax2.imshow(field_grid, extent=[0, 100, 0, 100], origin='lower',
           cmap='YlOrRd', alpha=0.5)

cells = voronoi.compute_voronoi(final_positions, field_grid)
for i, cell in enumerate(cells):
    ax2.scatter(final_positions[i, 0], final_positions[i, 1],
                c=[colors[i]], s=150, marker='o', edgecolors='black',
                linewidths=2, zorder=10)
    ax2.scatter(cell.centroid[0], cell.centroid[1],
                c=[colors[i]], s=80, marker='x', linewidths=2, zorder=9)
    ax2.plot([final_positions[i, 0], cell.centroid[0]],
             [final_positions[i, 1], cell.centroid[1]],
             color=colors[i], linestyle='--', alpha=0.7)
    ax2.annotate(f'UAV{i}', (final_positions[i, 0] + 2, final_positions[i, 1] + 2))

ax2.set_xlim(0, 100)
ax2.set_ylim(0, 100)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Voronoi分割与质心')
ax2.set_aspect('equal')

# 图3: 覆盖代价曲线
ax3 = axes[1, 0]
time_axis = np.arange(len(history_cost)) * dt
ax3.plot(time_axis, history_cost, 'b-', linewidth=2)
ax3.fill_between(time_axis, history_cost, alpha=0.2)
ax3.axhline(y=min(history_cost), color='g', linestyle='--',
            alpha=0.7, label=f'最小代价:  {min(history_cost):.1f}')
ax3.axhline(y=np.mean(history_cost), color='orange', linestyle=':',
            alpha=0.7, label=f'平均代价: {np.mean(history_cost):.1f}')
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('覆盖代价')
ax3.set_title('覆盖代价变化')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: 性能统计
ax4 = axes[1, 1]
metrics = ['初始代价', '最终代价', '最小代价', '平均代价']
values = [history_cost[0], history_cost[-1], min(history_cost), np.mean(history_cost)]
bar_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
bars = ax4.bar(metrics, values, color=bar_colors)

for bar, val in zip(bars, values):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10)

improvement = (history_cost[0] - history_cost[-1]) / history_cost[0] * 100
ax4.text(0.5, 0.95, f'总体改善率: {improvement:.1f}%',
         transform=ax4.transAxes, ha='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax4.set_ylabel('代价值')
ax4.set_title('性能统计')

plt.tight_layout()
plt.savefig('test_results.png', dpi=150, bbox_inches='tight')
print("\n结果图已保存至:  test_results.png")

print("\n显示图形窗口...")
plt.show()

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)