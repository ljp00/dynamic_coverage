"""
Control Barrier Function (CBF) 安全约束
确保无人机间安全距离和边界约束
"""
import numpy as np
from typing import List, Tuple
from scipy.optimize import minimize, LinearConstraint
import cvxpy as cp


class CBFSafetyFilter:
    """CBF安全滤波器"""

    def __init__(self, safe_distance: float = 3.0,
                 gamma: float = 1.0,
                 domain: Tuple[float, float, float, float] = None):
        """
        Args:
            safe_distance: 无人机间最小安全距离
            gamma: CBF class-K函数参数
            domain: 工作区域边界 (用于边界约束)
        """
        self.d_safe = safe_distance
        self.gamma = gamma
        self.domain = domain

    def collision_barrier(self, pi: np.ndarray, pj: np.ndarray) -> float:
        """
        碰撞避免障碍函数
        h(p_i, p_j) = ||p_i - p_j||² - d_safe²
        """
        return np.linalg.norm(pi - pj) ** 2.0 - self.d_safe **2.0

    def collision_barrier_gradient(self, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
        """碰撞障碍函数对p_i的梯度"""
        return 2 * (pi - pj)

    def boundary_barrier(self, p: np.ndarray) -> List[float]:
        """
        边界约束障碍函数
        返回四个边界的障碍值
        """
        if self.domain is None:
            return []
        x_min, x_max, y_min, y_max = self.domain
        margin = 1.0  # 边界余量
        return [
            p[0] - x_min - margin,  # 左边界
            x_max - p[0] - margin,  # 右边界
            p[1] - y_min - margin,  # 下边界
            y_max - p[1] - margin  # 上边界
        ]

    def filter_control(self, positions: np.ndarray,
                       nominal_controls: np.ndarray) -> np.ndarray:
        """
        CBF-QP安全滤波
        min ||u - u_nom||²
        s.t. ḣ + γh ≥ 0 (CBF约束)

        Args:
            positions: 所有无人机位置 shape (N, 2)
            nominal_controls: 标称控制输入 shape (N, 2)
        Returns:
            safe_controls: 安全控制输入 shape (N, 2)
        """
        num_agents = len(positions)
        safe_controls = np.zeros_like(nominal_controls)

        for i in range(num_agents):
            # 使用CVXPY求解QP
            u = cp.Variable(2)

            # 目标：最小化与标称控制的偏差
            objective = cp.Minimize(cp.sum_squares(u - nominal_controls[i]))

            constraints = []

            # 碰撞避免约束
            for j in range(num_agents):
                if i != j:
                    h = self.collision_barrier(positions[i], positions[j])
                    grad_h = self.collision_barrier_gradient(positions[i], positions[j])

                    # CBF约束:  grad_h @ u + gamma * h >= 0
                    constraints.append(grad_h @ u + self.gamma * h >= 0)

            # 边界约束
            if self.domain is not None:
                h_bounds = self.boundary_barrier(positions[i])
                # 边界梯度
                grad_bounds = [
                    np.array([1, 0]),  # 左
                    np.array([-1, 0]),  # 右
                    np.array([0, 1]),  # 下
                    np.array([0, -1])  # 上
                ]
                for h_b, grad_b in zip(h_bounds, grad_bounds):
                    constraints.append(grad_b @ u + self.gamma * h_b >= 0)

            # 求解QP
            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.OSQP, verbose=False)
                if problem.status == 'optimal':
                    safe_controls[i] = u.value
                else:
                    # 如果优化失败，使用保守策略
                    safe_controls[i] = nominal_controls[i] * 0.5
            except:
                safe_controls[i] = nominal_controls[i] * 0.5

        return safe_controls


class DistributedCBF(CBFSafetyFilter):
    """分布式CBF（仅考虑邻居）"""

    def __init__(self, communication_radius: float = 20.0, **kwargs):
        super().__init__(**kwargs)
        self.comm_radius = communication_radius

    def get_neighbors(self, positions: np.ndarray, agent_id: int) -> List[int]:
        """获取通信范围内的邻居"""
        neighbors = []
        for j in range(len(positions)):
            if j != agent_id:
                dist = np.linalg.norm(positions[agent_id] - positions[j])
                if dist < self.comm_radius:
                    neighbors.append(j)
        return neighbors

    def filter_control(self, positions: np.ndarray,
                       nominal_controls: np.ndarray) -> np.ndarray:
        """分布式CBF滤波（仅考虑邻居）"""
        num_agents = len(positions)
        safe_controls = np.zeros_like(nominal_controls)

        for i in range(num_agents):
            neighbors = self.get_neighbors(positions, i)

            u = cp.Variable(2)
            objective = cp.Minimize(cp.sum_squares(u - nominal_controls[i]))
            constraints = []

            # 仅对邻居添加碰撞约束
            for j in neighbors:
                h = self.collision_barrier(positions[i], positions[j])
                grad_h = self.collision_barrier_gradient(positions[i], positions[j])
                constraints.append(grad_h @ u + self.gamma * h >= 0)

            # 边界约束
            if self.domain is not None:
                h_bounds = self.boundary_barrier(positions[i])
                grad_bounds = [
                    np.array([1, 0]), np.array([-1, 0]),
                    np.array([0, 1]), np.array([0, -1])
                ]
                for h_b, grad_b in zip(h_bounds, grad_bounds):
                    constraints.append(grad_b @ u + self.gamma * h_b >= 0)

            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.OSQP, verbose=False)
                if problem.status == 'optimal':
                    safe_controls[i] = u.value
                else:
                    safe_controls[i] = nominal_controls[i] * 0.5
            except:
                safe_controls[i] = nominal_controls[i] * 0.5

        return safe_controls