"""
动态敏感度场模块
支持热点沿固定轨迹移动：线性、圆周、正弦曲线
"""
import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class MotionType(Enum):
    """热点运动类型"""
    LINEAR = "linear"  # 线性往返
    CIRCULAR = "circular"  # 圆周运动
    SINUSOIDAL = "sinusoidal"  # 正弦曲线
    STATIC = "static"  # 静止


@dataclass
class HotSpot:
    """动态热点"""
    center: np.ndarray  # 当前中心位置
    intensity: float  # 强度
    spread: float  # 扩散范围（标准差）

    # 运动参数
    motion_type: MotionType  # 运动类型
    origin: np.ndarray  # 运动起点/中心
    velocity: float  # 运动速度
    amplitude: np.ndarray  # 运动幅度 [x方向, y方向]
    phase: float  # 初始相位

    def update_position(self, time: float):
        """根据时间更新热点位置"""
        if self.motion_type == MotionType.STATIC:
            pass  # 静止不动

        elif self.motion_type == MotionType.LINEAR:
            # 线性往返运动:  origin + amplitude * sin(velocity * t + phase)
            self.center = self.origin + self.amplitude * np.sin(self.velocity * time + self.phase)

        elif self.motion_type == MotionType.CIRCULAR:
            # 圆周运动: origin + [A_x * cos(wt), A_y * sin(wt)]
            angle = self.velocity * time + self.phase
            self.center = self.origin + np.array([
                self.amplitude[0] * np.cos(angle),
                self.amplitude[1] * np.sin(angle)
            ])

        elif self.motion_type == MotionType.SINUSOIDAL:
            # 正弦曲线: x方向匀速，y方向正弦
            x_offset = self.amplitude[0] * np.sin(self.velocity * time + self.phase)
            y_offset = self.amplitude[1] * np.cos(self.velocity * 0.5 * time + self.phase)
            self.center = self.origin + np.array([x_offset, y_offset])


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

    def add_hotspot(self, center: np.ndarray,
                    intensity: float = 1.0,
                    spread: float = 10.0,
                    motion_type: MotionType = MotionType.STATIC,
                    velocity: float = 0.5,
                    amplitude: np.ndarray = None,
                    phase: float = 0.0):
        """
        添加热点

        Args:
            center: 初始中心位置 / 运动中心
            intensity: 强度
            spread: 扩散范围
            motion_type: 运动类型
            velocity: 运动角速度/速度
            amplitude: 运动幅度 [x, y]
            phase: 初始相位
        """
        if amplitude is None:
            amplitude = np.array([15.0, 15.0])

        self.hotspots.append(HotSpot(
            center=np.array(center, dtype=float),
            intensity=intensity,
            spread=spread,
            motion_type=motion_type,
            origin=np.array(center, dtype=float),
            velocity=velocity,
            amplitude=np.array(amplitude, dtype=float),
            phase=phase
        ))

    def add_preset_hotspots(self, preset: str = "mixed", seed: int = None):
        """
        添加预设热点配置

        Args:
            preset: 预设类型
                - "static": 静态热点
                - "linear":  线性移动热点
                - "circular": 圆周运动热点
                - "mixed": 混合运动热点
        """
        if seed is not None:
            np.random.seed(seed)

        x_min, x_max, y_min, y_max = self.domain
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        if preset == "static":
            # 静态热点
            self.add_hotspot([30, 30], intensity=1.5, spread=12, motion_type=MotionType.STATIC)
            self.add_hotspot([70, 70], intensity=2.0, spread=15, motion_type=MotionType.STATIC)
            self.add_hotspot([30, 70], intensity=1.0, spread=10, motion_type=MotionType.STATIC)
            self.add_hotspot([70, 30], intensity=1.2, spread=11, motion_type=MotionType.STATIC)

        elif preset == "linear":
            # 线性往返移动
            self.add_hotspot([cx, 30], intensity=2.0, spread=12,
                             motion_type=MotionType.LINEAR, velocity=0.3,
                             amplitude=[25, 0], phase=0)
            self.add_hotspot([cx, 70], intensity=1.5, spread=10,
                             motion_type=MotionType.LINEAR, velocity=0.4,
                             amplitude=[20, 0], phase=np.pi)
            self.add_hotspot([30, cy], intensity=1.8, spread=11,
                             motion_type=MotionType.LINEAR, velocity=0.35,
                             amplitude=[0, 20], phase=np.pi / 2)

        elif preset == "circular":
            # 圆周运动
            self.add_hotspot([cx, cy], intensity=2.0, spread=12,
                             motion_type=MotionType.CIRCULAR, velocity=0.2,
                             amplitude=[25, 25], phase=0)
            self.add_hotspot([30, 30], intensity=1.5, spread=10,
                             motion_type=MotionType.CIRCULAR, velocity=0.3,
                             amplitude=[15, 15], phase=np.pi / 2)
            self.add_hotspot([70, 70], intensity=1.8, spread=11,
                             motion_type=MotionType.CIRCULAR, velocity=0.25,
                             amplitude=[18, 18], phase=np.pi)

        elif preset == "mixed":
            # 混合运动
            self.add_hotspot([cx, cy], intensity=2.0, spread=15,
                             motion_type=MotionType.CIRCULAR, velocity=0.15,
                             amplitude=[20, 20], phase=0)
            self.add_hotspot([25, 50], intensity=1.5, spread=12,
                             motion_type=MotionType.LINEAR, velocity=0.25,
                             amplitude=[0, 25], phase=0)
            self.add_hotspot([75, 50], intensity=1.5, spread=12,
                             motion_type=MotionType.LINEAR, velocity=0.25,
                             amplitude=[0, 25], phase=np.pi)
            self.add_hotspot([50, 25], intensity=1.2, spread=10,
                             motion_type=MotionType.SINUSOIDAL, velocity=0.3,
                             amplitude=[20, 15], phase=np.pi / 4)
        else:
            # 默认：混合
            self.add_preset_hotspots("mixed", seed)

    def update(self, dt: float):
        """更新敏感度场"""
        self.time += dt
        for hotspot in self.hotspots:
            hotspot.update_position(self.time)
            # 确保热点在域内
            self._clamp_hotspot_position(hotspot)

    def _clamp_hotspot_position(self, hotspot: HotSpot):
        """限制热点位置在域内"""
        margin = hotspot.spread * 0.5
        hotspot.center[0] = np.clip(hotspot.center[0],
                                    self.domain[0] + margin,
                                    self.domain[1] - margin)
        hotspot.center[1] = np.clip(hotspot.center[1],
                                    self.domain[2] + margin,
                                    self.domain[3] - margin)

    def get_density(self, positions: np.ndarray) -> np.ndarray:
        """获取指定位置的敏感度值"""
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

    def get_hotspot_positions(self) -> np.ndarray:
        """获取所有热点当前位置"""
        return np.array([h.center for h in self.hotspots])