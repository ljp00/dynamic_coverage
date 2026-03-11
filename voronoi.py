"""
Voronoi分割与覆盖控制
"""
import numpy as np
from scipy.spatial import Voronoi
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class VoronoiCell:
    """Voronoi单元"""
    agent_id: int
    vertices: np.ndarray      # 顶点坐标
    centroid: np.ndarray      # 质心
    mass: float               # 质量（敏感度积分）
    area: float               # 面积


class WeightedVoronoi:
    """加权Voronoi分割"""

    def __init__(self, domain: Tuple[float, float, float, float],
                 resolution: int = 50):
        """
        Args:
            domain: (x_min, x_max, y_min, y_max)
            resolution: 积分分辨率
        """
        self.domain = domain
        self.resolution = resolution

        # 创建积分网格
        self.x_grid = np.linspace(domain[0], domain[1], resolution)
        self.y_grid = np.linspace(domain[2], domain[3], resolution)
        self.dx = self.x_grid[1] - self.x_grid[0]
        self.dy = self.y_grid[1] - self.y_grid[0]
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        self.grid_points = np.column_stack([self.X.ravel(), self.Y.ravel()])

    def compute_voronoi(self, positions: np.ndarray,
                        density_field: np.ndarray) -> List[VoronoiCell]:
        """
        计算加权Voronoi分割
        Args:
            positions: 无人机位置 shape (N, 2)
            density_field: 敏感度场网格值 shape (resolution, resolution)
        Returns:
            Voronoi单元列表
        """
        num_agents = len(positions)

        # 关键修复：确保 density_field 与网格分辨率匹配
        # 如果传入的 density_field 分辨率不同，进行插值
        if density_field. shape[0] != self.resolution or density_field.shape[1] != self.resolution:
            from scipy.ndimage import zoom
            zoom_factors = (self.resolution / density_field.shape[0],
                          self.resolution / density_field. shape[1])
            density_field = zoom(density_field, zoom_factors, order=1)

        density_flat = density_field.ravel()

        # 计算每个网格点到各无人机的距离
        distances = np.zeros((len(self.grid_points), num_agents))
        for i, pos in enumerate(positions):
            distances[:, i] = np.linalg.norm(self.grid_points - pos, axis=1)

        # 分配网格点到最近的无人机
        assignments = np.argmin(distances, axis=1)

        cells = []
        for i in range(num_agents):
            mask = assignments == i
            if not np.any(mask):
                # 如果没有分配到任何点，使用当前位置
                cells.append(VoronoiCell(
                    agent_id=i,
                    vertices=np.array([]),
                    centroid=positions[i]. copy(),
                    mass=0.0,
                    area=0.0
                ))
                continue

            # 获取该单元的网格点和密度
            cell_points = self.grid_points[mask]
            cell_density = density_flat[mask]

            # 计算加权质心
            total_mass = np.sum(cell_density) * self.dx * self.dy
            if total_mass > 1e-10:
                centroid = np.sum(cell_points * cell_density[: , None], axis=0) * \
                          self.dx * self.dy / total_mass
            else:
                centroid = positions[i].copy()

            # 计算面积
            area = np.sum(mask) * self.dx * self.dy

            cells.append(VoronoiCell(
                agent_id=i,
                vertices=cell_points,  # 简化：存储所有属于该单元的点
                centroid=centroid,
                mass=total_mass,
                area=area
            ))

        return cells

    def compute_coverage_cost(self, positions: np.ndarray,
                             density_field: np.ndarray) -> float:
        """
        计算覆盖代价函数
        H = Σ∫_{V_i} ||q - p_i||² * φ(q) dq
        """
        # 关键修复：确保 density_field 与网格分辨率匹配
        if density_field.shape[0] != self.resolution or density_field.shape[1] != self. resolution:
            from scipy.ndimage import zoom
            zoom_factors = (self.resolution / density_field.shape[0],
                          self.resolution / density_field.shape[1])
            density_field = zoom(density_field, zoom_factors, order=1)

        density_flat = density_field.ravel()
        num_agents = len(positions)

        # 计算距离
        distances = np.zeros((len(self.grid_points), num_agents))
        for i, pos in enumerate(positions):
            distances[:, i] = np. linalg.norm(self. grid_points - pos, axis=1)

        # 分配并计算代价
        assignments = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(distances)), assignments]

        cost = np.sum(min_distances ** 2 * density_flat) * self.dx * self.dy
        return cost