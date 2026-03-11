"""
动态敏感度场模块
支持多种时变场模型：高斯热点、扩散模型等
"""
import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class HotSpot:
    """动态热点"""
    center: np.ndarray  # 中心位置
    intensity: float  # 强度
    spread: float  # 扩散范围
    velocity: np.ndarray  # 移动速度


class DynamicSensitivityField:
    """动态敏感度场"""

    def __init__(self, domain: Tuple[float, float, float, float],
                 resolution: int = 50):
        """
        Args:
            domain: (x_min, x_max, y_min, y_max)
            resolution: 网格分辨率
        """
        self.domain = domain
        self.resolution = resolution
        self.hotspots: List[HotSpot] = []
        self.time = 0.0

        # 创建网格
        self.x_grid = np.linspace(domain[0], domain[1], resolution)
        self.y_grid = np.linspace(domain[2], domain[3], resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

    def add_hotspot(self, center: np.ndarray, intensity: float = 1.0,
                    spread: float = 10.0, velocity: np.ndarray = None):
        """添加动态热点"""
        if velocity is None:
            velocity = np.zeros(2)
        self.hotspots.append(HotSpot(
            center=np.array(center),
            intensity=intensity,
            spread=spread,
            velocity=np.array(velocity)
        ))

    def add_random_hotspots(self, num_hotspots: int = 3, seed: int = None):
        """添加随机热点"""
        if seed is not None:
            np.random.seed(seed)

        for _ in range(num_hotspots):
            center = np.array([
                np.random.uniform(self.domain[0] + 10, self.domain[1] - 10),
                np.random.uniform(self.domain[2] + 10, self.domain[3] - 10)
            ])
            intensity = np.random.uniform(0.5, 2.0)
            spread = np.random.uniform(8.0, 20.0)
            velocity = np.random.uniform(-1.0, 1.0, size=2)
            self.add_hotspot(center, intensity, spread, velocity)

    def update(self, dt: float):
        """更新敏感度场（时间演化）"""
        self.time += dt
        for hotspot in self.hotspots:
            # 更新热点位置
            hotspot.center += hotspot.velocity * dt

            # 边界反弹
            for i in range(2):
                if hotspot.center[i] < self.domain[i * 2] + 5:
                    hotspot.center[i] = self.domain[i * 2] + 5
                    hotspot.velocity[i] *= -1
                elif hotspot.center[i] > self.domain[i * 2 + 1] - 5:
                    hotspot.center[i] = self.domain[i * 2 + 1] - 5
                    hotspot.velocity[i] *= -1

            # 强度随时间变化（可选）
            hotspot.intensity *= (1.0 + 0.01 * np.sin(0.1 * self.time))

    def get_density(self, positions: np.ndarray) -> np.ndarray:
        """
        获取指定位置的敏感度值
        Args:
            positions: shape (N, 2) 或 (2,)
        Returns:
            敏感度值
        """
        positions = np.atleast_2d(positions)
        density = np.zeros(len(positions))

        for hotspot in self.hotspots:
            dist_sq = np.sum((positions - hotspot.center) ** 2, axis=1)
            density += hotspot.intensity * np.exp(-dist_sq / (2 * hotspot.spread ** 2))

        # 添加基础敏感度
        density += 0.1
        return density

    def get_field_grid(self) -> np.ndarray:
        """获取整个场的网格值"""
        positions = np.column_stack([self.X.ravel(), self.Y.ravel()])
        density = self.get_density(positions)
        return density.reshape(self.resolution, self.resolution)

    def get_gradient(self, position: np.ndarray, eps: float = 0.1) -> np.ndarray:
        """计算敏感度场梯度（数值微分）"""
        grad = np.zeros(2)
        for i in range(2):
            pos_plus = position.copy()
            pos_minus = position.copy()
            pos_plus[i] += eps
            pos_minus[i] -= eps
            grad[i] = (self.get_density(pos_plus)[0] -
                       self.get_density(pos_minus)[0]) / (2 * eps)
        return grad