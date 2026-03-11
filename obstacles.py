"""
障碍物环境模块
支持静态障碍物、动态障碍物、禁飞区等
"""
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class ObstacleType(Enum):
    """障碍物类型"""
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


@dataclass
class Obstacle(ABC):
    """障碍物基类"""
    id: int
    obstacle_type: ObstacleType
    is_static: bool = True
    safety_margin: float = 1.0  # 安全余量

    @abstractmethod
    def contains_point(self, point: np.ndarray) -> bool:
        """判断点是否在障碍物内"""
        pass

    @abstractmethod
    def distance_to_point(self, point: np.ndarray) -> float:
        """计算点到障碍物边界的距离（负值表示在内部）"""
        pass

    @abstractmethod
    def get_nearest_point(self, point: np.ndarray) -> np.ndarray:
        """获取障碍物边界上距离给定点最近的点"""
        pass

    @abstractmethod
    def update(self, dt: float):
        """更新障碍物状态（用于动态障碍物）"""
        pass

    @abstractmethod
    def get_vertices(self) -> np.ndarray:
        """获取障碍物顶点（用于绘图）"""
        pass


@dataclass
class CircleObstacle(Obstacle):
    """圆形障碍物"""
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 5.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # 动态障碍物速度

    def __post_init__(self):
        self.obstacle_type = ObstacleType.CIRCLE
        self.center = np.array(self.center, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)

    def contains_point(self, point: np.ndarray) -> bool:
        """判断点是否在圆内（考虑安全余量）"""
        dist = np.linalg.norm(point - self.center)
        return dist < (self.radius + self.safety_margin)

    def distance_to_point(self, point: np.ndarray) -> float:
        """点到圆边界的距离"""
        dist_to_center = np.linalg.norm(point - self.center)
        return dist_to_center - self.radius - self.safety_margin

    def get_nearest_point(self, point: np.ndarray) -> np.ndarray:
        """获取圆边界上最近点"""
        direction = point - self.center
        dist = np.linalg.norm(direction)
        if dist < 1e-10:
            # 点在圆心，返回任意边界点
            return self.center + np.array([self.radius, 0])
        return self.center + direction / dist * self.radius

    def update(self, dt: float):
        """更新位置（动态障碍物）"""
        if not self.is_static:
            self.center += self.velocity * dt

    def get_vertices(self, num_points: int = 32) -> np.ndarray:
        """获取圆的近似顶点"""
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        vertices = np.column_stack([
            self.center[0] + self.radius * np.cos(angles),
            self.center[1] + self.radius * np.sin(angles)
        ])
        return vertices


@dataclass
class RectangleObstacle(Obstacle):
    """矩形障碍物"""
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    width: float = 10.0
    height: float = 10.0
    angle: float = 0.0  # 旋转角度（弧度）
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    angular_velocity: float = 0.0  # 角速度

    def __post_init__(self):
        self.obstacle_type = ObstacleType.RECTANGLE
        self.center = np.array(self.center, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)
        self._update_corners()

    def _update_corners(self):
        """更新角点位置"""
        hw, hh = self.width / 2, self.height / 2
        # 局部坐标系下的角点
        local_corners = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh],
            [-hw, hh]
        ])
        # 旋转矩阵
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        # 全局坐标
        self.corners = (rotation @ local_corners.T).T + self.center

    def contains_point(self, point: np.ndarray) -> bool:
        """判断点是否在矩形内"""
        # 转换到局部坐标系
        local_point = self._to_local(point)
        hw = self.width / 2 + self.safety_margin
        hh = self.height / 2 + self.safety_margin
        return (abs(local_point[0]) < hw) and (abs(local_point[1]) < hh)

    def _to_local(self, point: np.ndarray) -> np.ndarray:
        """转换到局部坐标系"""
        cos_a, sin_a = np.cos(-self.angle), np.sin(-self.angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return rotation @ (point - self.center)

    def _to_global(self, local_point: np.ndarray) -> np.ndarray:
        """转换到全局坐标系"""
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return rotation @ local_point + self.center

    def distance_to_point(self, point: np.ndarray) -> float:
        """点到矩形边界的距离"""
        local_point = self._to_local(point)
        hw = self.width / 2 + self.safety_margin
        hh = self.height / 2 + self.safety_margin

        # 计算到矩形的距离
        dx = max(abs(local_point[0]) - hw, 0)
        dy = max(abs(local_point[1]) - hh, 0)

        outside_dist = np.sqrt(dx ** 2 + dy ** 2)

        if outside_dist > 0:
            return outside_dist
        else:
            # 点在矩形内部，返回负距离
            return -min(hw - abs(local_point[0]), hh - abs(local_point[1]))

    def get_nearest_point(self, point: np.ndarray) -> np.ndarray:
        """获取矩形边界上最近点"""
        local_point = self._to_local(point)
        hw, hh = self.width / 2, self.height / 2

        # 将点裁剪到矩形边界
        nearest_local = np.array([
            np.clip(local_point[0], -hw, hw),
            np.clip(local_point[1], -hh, hh)
        ])

        # 如果点在内部，投影到最近边
        if self.contains_point(point):
            dist_to_edges = [
                hw - abs(local_point[0]),  # 到左/右边
                hh - abs(local_point[1])  # 到上/下边
            ]
            if dist_to_edges[0] < dist_to_edges[1]:
                nearest_local[0] = hw * np.sign(local_point[0])
            else:
                nearest_local[1] = hh * np.sign(local_point[1])

        return self._to_global(nearest_local)

    def update(self, dt: float):
        """更新状态"""
        if not self.is_static:
            self.center += self.velocity * dt
            self.angle += self.angular_velocity * dt
            self._update_corners()

    def get_vertices(self) -> np.ndarray:
        """获取矩形顶点"""
        self._update_corners()
        return self.corners


@dataclass
class PolygonObstacle(Obstacle):
    """多边形障碍物"""
    vertices: np.ndarray = field(default_factory=lambda: np.array([[0, 0], [1, 0], [0.5, 1]]))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        self.obstacle_type = ObstacleType.POLYGON
        self.vertices = np.array(self.vertices, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)
        self._compute_center()

    def _compute_center(self):
        """计算多边形中心"""
        self.center = np.mean(self.vertices, axis=0)

    def contains_point(self, point: np.ndarray) -> bool:
        """使用射线法判断点是否在多边形内"""
        n = len(self.vertices)
        inside = False

        j = n - 1
        for i in range(n):
            vi, vj = self.vertices[i], self.vertices[j]
            if ((vi[1] > point[1]) != (vj[1] > point[1])) and \
                    (point[0] < (vj[0] - vi[0]) * (point[1] - vi[1]) / (vj[1] - vi[1]) + vi[0]):
                inside = not inside
            j = i

        return inside

    def distance_to_point(self, point: np.ndarray) -> float:
        """点到多边形边界的距离"""
        min_dist = float('inf')
        n = len(self.vertices)

        for i in range(n):
            v1, v2 = self.vertices[i], self.vertices[(i + 1) % n]
            dist = self._point_to_segment_distance(point, v1, v2)
            min_dist = min(min_dist, dist)

        if self.contains_point(point):
            return -min_dist - self.safety_margin
        return min_dist - self.safety_margin

    def _point_to_segment_distance(self, point: np.ndarray,
                                   v1: np.ndarray, v2: np.ndarray) -> float:
        """计算点到线段的距离"""
        segment = v2 - v1
        length_sq = np.dot(segment, segment)

        if length_sq < 1e-10:
            return np.linalg.norm(point - v1)

        t = max(0, min(1, np.dot(point - v1, segment) / length_sq))
        projection = v1 + t * segment
        return np.linalg.norm(point - projection)

    def get_nearest_point(self, point: np.ndarray) -> np.ndarray:
        """获取多边形边界上最近点"""
        min_dist = float('inf')
        nearest = self.vertices[0].copy()
        n = len(self.vertices)

        for i in range(n):
            v1, v2 = self.vertices[i], self.vertices[(i + 1) % n]
            segment = v2 - v1
            length_sq = np.dot(segment, segment)

            if length_sq < 1e-10:
                candidate = v1
            else:
                t = max(0, min(1, np.dot(point - v1, segment) / length_sq))
                candidate = v1 + t * segment

            dist = np.linalg.norm(point - candidate)
            if dist < min_dist:
                min_dist = dist
                nearest = candidate.copy()

        return nearest

    def update(self, dt: float):
        """更新位置"""
        if not self.is_static:
            self.vertices += self.velocity * dt
            self._compute_center()

    def get_vertices(self) -> np.ndarray:
        """获取多边形顶点"""
        return self.vertices


@dataclass
class NoFlyZone:
    """禁飞区"""
    id: int
    center: np.ndarray
    radius: float
    active: bool = True
    time_window: Tuple[float, float] = (0, float('inf'))  # 激活时间窗口
    penalty: float = 100.0  # 进入禁飞区的惩罚

    def __post_init__(self):
        self.center = np.array(self.center, dtype=float)

    def is_active(self, current_time: float) -> bool:
        """判断禁飞区是否激活"""
        return self.active and (self.time_window[0] <= current_time <= self.time_window[1])

    def contains_point(self, point: np.ndarray) -> bool:
        """判断点是否在禁飞区内"""
        return np.linalg.norm(point - self.center) < self.radius

    def distance_to_point(self, point: np.ndarray) -> float:
        """点到禁飞区边界的距离"""
        return np.linalg.norm(point - self.center) - self.radius


class ObstacleEnvironment:
    """障碍物环境管理器"""

    def __init__(self, domain: Tuple[float, float, float, float]):
        """
        Args:
            domain: 工作区域 (x_min, x_max, y_min, y_max)
        """
        self.domain = domain
        self.obstacles: List[Obstacle] = []
        self.no_fly_zones: List[NoFlyZone] = []
        self._obstacle_counter = 0
        self._nfz_counter = 0

    def add_circle_obstacle(self, center: np.ndarray, radius: float,
                            is_static: bool = True, velocity: np.ndarray = None,
                            safety_margin: float = 1.0) -> int:
        """添加圆形障碍物"""
        obs = CircleObstacle(
            id=self._obstacle_counter,
            center=center,
            radius=radius,
            is_static=is_static,
            velocity=velocity if velocity is not None else np.zeros(2),
            safety_margin=safety_margin
        )
        self.obstacles.append(obs)
        self._obstacle_counter += 1
        return obs.id

    def add_rectangle_obstacle(self, center: np.ndarray, width: float, height: float,
                               angle: float = 0.0, is_static: bool = True,
                               velocity: np.ndarray = None,
                               safety_margin: float = 1.0) -> int:
        """添加矩形障碍物"""
        obs = RectangleObstacle(
            id=self._obstacle_counter,
            center=center,
            width=width,
            height=height,
            angle=angle,
            is_static=is_static,
            velocity=velocity if velocity is not None else np.zeros(2),
            safety_margin=safety_margin
        )
        self.obstacles.append(obs)
        self._obstacle_counter += 1
        return obs.id

    def add_polygon_obstacle(self, vertices: np.ndarray,
                             is_static: bool = True,
                             velocity: np.ndarray = None,
                             safety_margin: float = 1.0) -> int:
        """添加多边形障碍物"""
        obs = PolygonObstacle(
            id=self._obstacle_counter,
            vertices=vertices,
            is_static=is_static,
            velocity=velocity if velocity is not None else np.zeros(2),
            safety_margin=safety_margin
        )
        self.obstacles.append(obs)
        self._obstacle_counter += 1
        return obs.id

    def add_no_fly_zone(self, center: np.ndarray, radius: float,
                        time_window: Tuple[float, float] = None,
                        penalty: float = 100.0) -> int:
        """添加禁飞区"""
        nfz = NoFlyZone(
            id=self._nfz_counter,
            center=center,
            radius=radius,
            time_window=time_window if time_window else (0, float('inf')),
            penalty=penalty
        )
        self.no_fly_zones.append(nfz)
        self._nfz_counter += 1
        return nfz.id

    def add_random_obstacles(self, num_obstacles: int = 5,
                             min_radius: float = 3.0,
                             max_radius: float = 8.0,
                             seed: int = None):
        """添加随机障碍物"""
        if seed is not None:
            np.random.seed(seed)

        margin = max_radius + 5
        for _ in range(num_obstacles):
            center = np.array([
                np.random.uniform(self.domain[0] + margin, self.domain[1] - margin),
                np.random.uniform(self.domain[2] + margin, self.domain[3] - margin)
            ])
            radius = np.random.uniform(min_radius, max_radius)

            # 确保不与现有障碍物重叠
            valid = True
            for obs in self.obstacles:
                if isinstance(obs, CircleObstacle):
                    if np.linalg.norm(center - obs.center) < radius + obs.radius + 2:
                        valid = False
                        break

            if valid:
                self.add_circle_obstacle(center, radius)

    def add_boundary_walls(self, wall_thickness: float = 2.0):
        """添加边界墙"""
        x_min, x_max, y_min, y_max = self.domain

        # 四面墙
        # 下墙
        self.add_rectangle_obstacle(
            center=np.array([(x_min + x_max) / 2, y_min - wall_thickness / 2]),
            width=x_max - x_min + 2 * wall_thickness,
            height=wall_thickness
        )
        # 上墙
        self.add_rectangle_obstacle(
            center=np.array([(x_min + x_max) / 2, y_max + wall_thickness / 2]),
            width=x_max - x_min + 2 * wall_thickness,
            height=wall_thickness
        )
        # 左墙
        self.add_rectangle_obstacle(
            center=np.array([x_min - wall_thickness / 2, (y_min + y_max) / 2]),
            width=wall_thickness,
            height=y_max - y_min
        )
        # 右墙
        self.add_rectangle_obstacle(
            center=np.array([x_max + wall_thickness / 2, (y_min + y_max) / 2]),
            width=wall_thickness,
            height=y_max - y_min
        )

    def update(self, dt: float):
        """更新所有动态障碍物"""
        for obs in self.obstacles:
            obs.update(dt)

            # 边界反弹（对于动态障碍物）
            if not obs.is_static and isinstance(obs, CircleObstacle):
                # 检查边界
                if obs.center[0] - obs.radius < self.domain[0]:
                    obs.center[0] = self.domain[0] + obs.radius
                    obs.velocity[0] *= -1
                elif obs.center[0] + obs.radius > self.domain[1]:
                    obs.center[0] = self.domain[1] - obs.radius
                    obs.velocity[0] *= -1

                if obs.center[1] - obs.radius < self.domain[2]:
                    obs.center[1] = self.domain[2] + obs.radius
                    obs.velocity[1] *= -1
                elif obs.center[1] + obs.radius > self.domain[3]:
                    obs.center[1] = self.domain[3] - obs.radius
                    obs.velocity[1] *= -1

    def is_collision(self, point: np.ndarray) -> bool:
        """检查点是否与任何障碍物碰撞"""
        for obs in self.obstacles:
            if obs.contains_point(point):
                return True
        return False

    def is_in_no_fly_zone(self, point: np.ndarray, current_time: float = 0) -> bool:
        """检查点是否在禁飞区内"""
        for nfz in self.no_fly_zones:
            if nfz.is_active(current_time) and nfz.contains_point(point):
                return True
        return False

    def get_min_obstacle_distance(self, point: np.ndarray) -> Tuple[float, Optional[Obstacle]]:
        """获取点到最近障碍物的距离"""
        min_dist = float('inf')
        nearest_obs = None

        for obs in self.obstacles:
            dist = obs.distance_to_point(point)
            if dist < min_dist:
                min_dist = dist
                nearest_obs = obs

        return min_dist, nearest_obs

    def get_obstacle_gradient(self, point: np.ndarray, eps: float = 0.1) -> np.ndarray:
        """计算障碍物距离场的梯度（指向远离障碍物的方向）"""
        grad = np.zeros(2)

        for i in range(2):
            point_plus = point.copy()
            point_minus = point.copy()
            point_plus[i] += eps
            point_minus[i] -= eps

            dist_plus, _ = self.get_min_obstacle_distance(point_plus)
            dist_minus, _ = self.get_min_obstacle_distance(point_minus)

            grad[i] = (dist_plus - dist_minus) / (2 * eps)

        return grad

    def get_repulsive_force(self, point: np.ndarray,
                            influence_distance: float = 10.0,
                            gain: float = 1.0) -> np.ndarray:
        """
        计算障碍物排斥力（用于人工势场法）

        F_rep = gain * (1/d - 1/d0) * (1/d^2) * grad(d)  当 d < d0
              = 0                                         当 d >= d0
        """
        min_dist, nearest_obs = self.get_min_obstacle_distance(point)

        if min_dist >= influence_distance or nearest_obs is None:
            return np.zeros(2)

        if min_dist < 0.1:
            min_dist = 0.1  # 避免除零

        # 计算梯度方向
        nearest_point = nearest_obs.get_nearest_point(point)
        direction = point - nearest_point
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            direction = np.random.randn(2)
            dir_norm = np.linalg.norm(direction)
        direction = direction / dir_norm

        # 排斥力大小
        magnitude = gain * (1.0 / min_dist - 1.0 / influence_distance) * (1.0 / min_dist ** 2)

        return magnitude * direction

    def is_path_clear(self, start: np.ndarray, end: np.ndarray,
                      num_checks: int = 20) -> bool:
        """检查路径是否无碰撞"""
        for t in np.linspace(0, 1, num_checks):
            point = start + t * (end - start)
            if self.is_collision(point):
                return False
        return True

    def get_occupancy_grid(self, resolution: int = 100) -> np.ndarray:
        """
        生成占据栅格地图

        Returns:
            grid: shape (resolution, resolution), 1表示障碍物，0表示自由空间
        """
        x_grid = np.linspace(self.domain[0], self.domain[1], resolution)
        y_grid = np.linspace(self.domain[2], self.domain[3], resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        grid = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                point = np.array([X[i, j], Y[i, j]])
                if self.is_collision(point):
                    grid[i, j] = 1

        return grid

    def get_distance_field(self, resolution: int = 100) -> np.ndarray:
        """
        生成距离场（到最近障碍物的距离）

        Returns:
            distance_field: shape (resolution, resolution)
        """
        x_grid = np.linspace(self.domain[0], self.domain[1], resolution)
        y_grid = np.linspace(self.domain[2], self.domain[3], resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        distance_field = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                point = np.array([X[i, j], Y[i, j]])
                dist, _ = self.get_min_obstacle_distance(point)
                distance_field[i, j] = dist

        return distance_field


class ObstacleCBF:
    """障碍物避免的CBF约束"""

    def __init__(self, obstacle_env: ObstacleEnvironment,
                 gamma: float = 1.0,
                 safety_margin: float = 1.0):
        """
        Args:
            obstacle_env: 障碍物环境
            gamma: CBF class-K函数参数
            safety_margin: 额外安全余量
        """
        self.env = obstacle_env
        self.gamma = gamma
        self.safety_margin = safety_margin

    def barrier_function(self, position: np.ndarray) -> float:
        """
        障碍物避免障碍函数
        h(x) = d(x) - d_safe > 0 表示安全
        """
        min_dist, _ = self.env.get_min_obstacle_distance(position)
        return min_dist - self.safety_margin

    def barrier_gradient(self, position: np.ndarray) -> np.ndarray:
        """障碍函数梯度"""
        return self.env.get_obstacle_gradient(position)

    def get_constraint(self, position: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        获取CBF约束:  grad_h @ u + gamma * h >= 0

        Returns:
            A: 约束矩阵行 (grad_h)
            b: 约束下界 (-gamma * h)
        """
        h = self.barrier_function(position)
        grad_h = self.barrier_gradient(position)

        return grad_h, -self.gamma * h


def create_maze_environment(domain: Tuple[float, float, float, float],
                            complexity: int = 3) -> ObstacleEnvironment:
    """
    创建迷宫环境

    Args:
        domain: 工作区域
        complexity: 复杂度等级 (1-5)
    """
    env = ObstacleEnvironment(domain)

    x_min, x_max, y_min, y_max = domain
    width = x_max - x_min
    height = y_max - y_min

    # 根据复杂度添加障碍物
    num_obstacles = complexity * 3

    for i in range(num_obstacles):
        # 随机选择障碍物类型
        obs_type = np.random.choice(['circle', 'rectangle'])

        center = np.array([
            np.random.uniform(x_min + 15, x_max - 15),
            np.random.uniform(y_min + 15, y_max - 15)
        ])

        if obs_type == 'circle':
            radius = np.random.uniform(3, 8)
            env.add_circle_obstacle(center, radius)
        else:
            w = np.random.uniform(5, 15)
            h = np.random.uniform(5, 15)
            angle = np.random.uniform(0, np.pi)
            env.add_rectangle_obstacle(center, w, h, angle)

    return env


def create_dynamic_environment(domain: Tuple[float, float, float, float],
                               num_static: int = 3,
                               num_dynamic: int = 2) -> ObstacleEnvironment:
    """
    创建包含动态障碍物的环境
    """
    env = ObstacleEnvironment(domain)

    x_min, x_max, y_min, y_max = domain

    # 静态障碍物
    for _ in range(num_static):
        center = np.array([
            np.random.uniform(x_min + 15, x_max - 15),
            np.random.uniform(y_min + 15, y_max - 15)
        ])
        radius = np.random.uniform(4, 7)
        env.add_circle_obstacle(center, radius, is_static=True)

    # 动态障碍物
    for _ in range(num_dynamic):
        center = np.array([
            np.random.uniform(x_min + 20, x_max - 20),
            np.random.uniform(y_min + 20, y_max - 20)
        ])
        radius = np.random.uniform(3, 5)
        velocity = np.random.uniform(-2, 2, size=2)
        env.add_circle_obstacle(center, radius, is_static=False, velocity=velocity)

    return env