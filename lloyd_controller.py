"""
Lloyd算法覆盖控制器
"""
import numpy as np
from typing import Tuple, Optional
from . voronoi import WeightedVoronoi, VoronoiCell


class LloydController:
    """基于Lloyd算法的覆盖控制器"""

    def __init__(self, domain: Tuple[float, float, float, float],
                 gain: float = 1.0,
                 max_velocity:  float = 5.0,
                 resolution: int = 50):
        """
        Args:
            domain: 工作区域
            gain: 控制增益
            max_velocity: 最大速度限制
            resolution:  Voronoi计算分辨率
        """
        self.domain = domain
        self.gain = gain
        self.max_velocity = max_velocity
        self.resolution = resolution
        self.voronoi = WeightedVoronoi(domain, resolution=resolution)

    def compute_control(self, positions: np.ndarray,
                       density_field: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        计算Lloyd控制输入
        u_i = k * (c_i - p_i)  质心方向

        Args:
            positions: 无人机位置 shape (N, 2)
            density_field: 敏感度场 (任意分辨率，会自动适配)
        Returns:
            velocities: 控制速度 shape (N, 2)
            cells: Voronoi单元信息
        """
        # 计算Voronoi分割（voronoi.py 内部会处理分辨率匹配）
        cells = self.voronoi.compute_voronoi(positions, density_field)

        # 计算控制输入
        velocities = np.zeros_like(positions)
        for i, cell in enumerate(cells):
            # 向质心移动
            velocities[i] = self.gain * (cell. centroid - positions[i])

        # 速度限制
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-10)  # 避免除零
        scale = np.minimum(1.0, self.max_velocity / speeds)
        velocities *= scale

        return velocities, cells

    def compute_predictive_control(self, positions: np.ndarray,
                                   current_density: np.ndarray,
                                   predicted_density: np.ndarray,
                                   alpha: float = 0.5) -> np.ndarray:
        """
        预测性覆盖控制（结合GP预测）

        Args:
            positions: 当前位置
            current_density: 当前敏感度场
            predicted_density: 预测的未来敏感度场
            alpha:  当前与预测的权重 (0-1)
        Returns:
            控制速度
        """
        # 计算当前场的控制
        vel_current, _ = self.compute_control(positions, current_density)

        # 计算预测场的控制
        vel_predicted, _ = self.compute_control(positions, predicted_density)

        # 加权组合
        velocities = alpha * vel_current + (1 - alpha) * vel_predicted

        # 速度限制
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-10)
        scale = np.minimum(1.0, self.max_velocity / speeds)
        velocities *= scale

        return velocities