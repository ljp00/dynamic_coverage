"""
模型预测控制 (MPC) 模块
结合覆盖目标和安全约束
"""
import numpy as np
from typing import Tuple, List, Optional
import cvxpy as cp


class CoverageMPC:
    """覆盖控制MPC"""

    def __init__(self, horizon: int = 10,
                 dt: float = 0.1,
                 max_velocity: float = 5.0,
                 max_acceleration: float = 2.0,
                 safe_distance: float = 3.0,
                 Q: np.ndarray = None,
                 R: np.ndarray = None):
        """
        Args:
            horizon: 预测时域
            dt: 时间步长
            max_velocity: 最大速度
            max_acceleration: 最大加速度
            safe_distance: 安全距离
            Q: 状态权重矩阵
            R: 控制权重矩阵
        """
        self.N = horizon
        self.dt = dt
        self.v_max = max_velocity
        self.a_max = max_acceleration
        self.d_safe = safe_distance
        self.Q = Q if Q is not None else np.eye(2)
        self.R = R if R is not None else 0.1 * np.eye(2)

    def solve(self, current_pos: np.ndarray,
              current_vel: np.ndarray,
              target_trajectory: np.ndarray,
              obstacle_positions: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解MPC优化问题

        Args:
            current_pos: 当前位置 (2,)
            current_vel: 当前速度 (2,)
            target_trajectory: 目标轨迹 shape (N, 2)
            obstacle_positions: 其他无人机位置 shape (M, 2)
        Returns:
            optimal_control: 最优控制序列 shape (N, 2)
            predicted_trajectory: 预测轨迹 shape (N+1, 2)
        """
        # 决策变量
        pos = cp.Variable((self.N + 1, 2))  # 位置
        vel = cp.Variable((self.N + 1, 2))  # 速度
        acc = cp.Variable((self.N, 2))  # 加速度（控制输入）

        # 初始条件约束
        constraints = [
            pos[0] == current_pos,
            vel[0] == current_vel
        ]

        # 动力学约束（双积分器模型）
        for k in range(self.N):
            constraints.append(
                pos[k + 1] == pos[k] + vel[k] * self.dt + 0.5 * acc[k] * self.dt ** 2
            )
            constraints.append(
                vel[k + 1] == vel[k] + acc[k] * self.dt
            )

        # 速度约束
        for k in range(self.N + 1):
            constraints.append(cp.norm(vel[k]) <= self.v_max)

        # 加速度约束
        for k in range(self.N):
            constraints.append(cp.norm(acc[k]) <= self.a_max)

        # 目标函数：跟踪目标轨迹 + 控制平滑
        cost = 0
        for k in range(self.N):
            # 位置跟踪代价
            cost += cp.quad_form(pos[k + 1] - target_trajectory[k], self.Q)
            # 控制代价
            cost += cp.quad_form(acc[k], self.R)

        # 求解
        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status == 'optimal':
                return acc.value, pos.value
            else:
                # 返回零控制
                return np.zeros((self.N, 2)), np.tile(current_pos, (self.N + 1, 1))
        except:
            return np.zeros((self.N, 2)), np.tile(current_pos, (self.N + 1, 1))


class MultiAgentMPC:
    """多智能体MPC（集中式）"""

    def __init__(self, num_agents: int, **kwargs):
        self.num_agents = num_agents
        self.single_mpc = CoverageMPC(**kwargs)
        self.N = self.single_mpc.N
        self.dt = self.single_mpc.dt
        self.d_safe = self.single_mpc.d_safe

    def solve_centralized(self, positions: np.ndarray,
                          velocities: np.ndarray,
                          target_trajectories: np.ndarray) -> np.ndarray:
        """
        集中式多智能体MPC（带碰撞约束）

        注意：这是简化版本，大规模问题建议使用分布式方法

        Args:
            positions: 所有无人机位置 shape (N_agents, 2)
            velocities: 所有无人机速度 shape (N_agents, 2)
            target_trajectories: 目标轨迹 shape (N_agents, horizon, 2)
        Returns:
            controls: 控制输入 shape (N_agents, 2) 仅第一步
        """
        N = self.num_agents
        H = self.N

        # 简化：依次求解每个智能体的MPC，前面智能体的轨迹作为约束
        all_trajectories = []
        all_controls = []

        for i in range(N):
            # 其他智能体的预测位置（已求解的使用轨迹，未求解的假设静止）
            other_trajectories = []
            for j in range(N):
                if j < i:
                    other_trajectories.append(all_trajectories[j])
                elif j > i:
                    # 假设静止
                    static_traj = np.tile(positions[j], (H + 1, 1))
                    other_trajectories.append(static_traj)

            # 求解单智能体MPC
            acc, traj = self.single_mpc.solve(
                positions[i], velocities[i], target_trajectories[i]
            )

            all_trajectories.append(traj)
            all_controls.append(acc[0] if len(acc) > 0 else np.zeros(2))

        return np.array(all_controls)