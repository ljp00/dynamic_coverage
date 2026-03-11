"""
改进的追踪控制器
解决超调问题：位置追踪 + 速度匹配
"""
import numpy as np
from typing import Tuple, Optional


class ImprovedTrackingController:
    """
    改进的追踪控制器

    核心改进：
    1. 不用随机采样目标点，直接追踪预测热点
    2. 加入速度前馈，匹配热点运动速度
    3. 使用PD控制，避免超调
    4. 智能体围绕热点形成编队，而非全部挤在热点
    """

    def __init__(self, domain: Tuple = (0, 100, 0, 100),
                 max_velocity: float = 5.0,
                 position_gain: float = 1.0,      # 位置误差增益 Kp
                 velocity_gain: float = 0.8,      # 速度前馈增益 Kv
                 damping_gain: float = 0.3,       # 阻尼增益 Kd
                 formation_radius: float = 12.0,  # 编队半径
                 separation_gain: float = 2.0,    # 分离增益
                 separation_radius: float = 8.0): # 分离半径

        self.domain = domain
        self.max_velocity = max_velocity
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.damping_gain = damping_gain
        self.formation_radius = formation_radius
        self.separation_gain = separation_gain
        self.separation_radius = separation_radius

        # 热点运动估计
        self.last_hotspot_pos = None
        self.last_time = None
        self.estimated_hotspot_velocity = np.zeros(2)

        # 智能体上一步速度（用于阻尼）
        self.last_velocities = None

    def compute_control(self,
                        positions: np.ndarray,
                        current_hotspot: np.ndarray,
                        predicted_hotspot: np.ndarray,
                        current_time: float,
                        prediction_dt: float = 1.0) -> np.ndarray:
        """
        计算追踪控制

        Args:
            positions: 智能体位置 [N x 2]
            current_hotspot: 当前热点位置
            predicted_hotspot: 预测的未来热点位置
            current_time: 当前时间
            prediction_dt: 预测的时间步长
        """
        n_agents = len(positions)
        velocities = np.zeros_like(positions)

        # 1. 估计热点运动速度
        self._estimate_hotspot_velocity(current_hotspot, current_time)

        # 2. 计算目标点：围绕预测热点的编队位置
        target_positions = self._compute_formation_targets(
            predicted_hotspot, n_agents
        )

        # 3. 为每个智能体计算控制
        for i in range(n_agents):
            # 位置误差
            pos_error = target_positions[i] - positions[i]

            # 位置控制（P控制）
            vel_position = self.position_gain * pos_error

            # 速度前馈（匹配热点速度，避免滞后）
            vel_feedforward = self.velocity_gain * self.estimated_hotspot_velocity

            # 阻尼（减少震荡）
            if self.last_velocities is not None:
                vel_damping = -self.damping_gain * self.last_velocities[i]
            else:
                vel_damping = np.zeros(2)

            # 合成控制
            velocities[i] = vel_position + vel_feedforward + vel_damping

        # 4. 添加分离力（防止碰撞）
        velocities += self._compute_separation(positions)

        # 5. 边界约束
        velocities += self._compute_boundary_force(positions)

        # 6. 速度限幅
        velocities = self._limit_velocity(velocities)

        # 记录当前速度（用于下一步阻尼计算）
        self.last_velocities = velocities.copy()

        return velocities

    def _estimate_hotspot_velocity(self, current_hotspot: np.ndarray, current_time: float):
        """估计热点运动速度"""
        if self.last_hotspot_pos is not None and self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0.01:
                # 一阶低通滤波估计速度
                raw_velocity = (current_hotspot - self.last_hotspot_pos) / dt
                alpha = 0.3  # 滤波系数
                self.estimated_hotspot_velocity = (
                    alpha * raw_velocity +
                    (1 - alpha) * self.estimated_hotspot_velocity
                )

        self.last_hotspot_pos = current_hotspot.copy()
        self.last_time = current_time

    def _compute_formation_targets(self, hotspot: np.ndarray, n_agents: int) -> np.ndarray:
        """
        计算编队目标位置
        智能体围绕热点均匀分布在圆周上
        """
        targets = np.zeros((n_agents, 2))

        for i in range(n_agents):
            angle = 2 * np.pi * i / n_agents
            offset = self.formation_radius * np.array([np.cos(angle), np.sin(angle)])
            targets[i] = hotspot + offset

        return targets

    def _compute_separation(self, positions: np.ndarray) -> np.ndarray:
        """计算分离力"""
        n_agents = len(positions)
        separation = np.zeros_like(positions)

        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)

                    if 0.1 < dist < self.separation_radius:
                        force = self.separation_gain * (1 - dist / self.separation_radius)
                        separation[i] += force * diff / dist

        return separation

    def _compute_boundary_force(self, positions: np.ndarray) -> np.ndarray:
        """边界约束力"""
        boundary_force = np.zeros_like(positions)
        margin, strength = 8.0, 3.0

        for i, pos in enumerate(positions):
            if pos[0] < self.domain[0] + margin:
                boundary_force[i, 0] += strength * (self.domain[0] + margin - pos[0])
            if pos[0] > self.domain[1] - margin:
                boundary_force[i, 0] -= strength * (pos[0] - (self.domain[1] - margin))
            if pos[1] < self.domain[2] + margin:
                boundary_force[i, 1] += strength * (self.domain[2] + margin - pos[1])
            if pos[1] > self.domain[3] - margin:
                boundary_force[i, 1] -= strength * (pos[1] - (self.domain[3] - margin))

        return boundary_force

    def _limit_velocity(self, velocities: np.ndarray) -> np.ndarray:
        """速度限幅"""
        for i in range(len(velocities)):
            speed = np.linalg.norm(velocities[i])
            if speed > self.max_velocity:
                velocities[i] = velocities[i] / speed * self.max_velocity
        return velocities


class PredictiveFormationController:
    """
    预测编队控制器

    结合：
    1. 运动模型预测热点位置
    2. 编队控制围绕热点
    3. 速度匹配避免超调
    """

    def __init__(self, domain=(0, 100, 0, 100), max_velocity=5.0, formation_radius=12.0):
        self.domain = domain
        self.max_velocity = max_velocity
        self.formation_radius = formation_radius

        # 追踪控制器
        self.tracking_controller = ImprovedTrackingController(
            domain=domain,
            max_velocity=max_velocity,
            position_gain=1.2,
            velocity_gain=0.8,
            damping_gain=0.2,
            formation_radius=formation_radius
        )

        # 热点位置历史（用于预测）
        self.hotspot_history = []
        self.time_history = []
        self.max_history = 20

    def update_hotspot(self, hotspot_pos: np.ndarray, current_time: float):
        """更新热点位置历史"""
        self.hotspot_history.append(hotspot_pos.copy())
        self.time_history.append(current_time)

        if len(self.hotspot_history) > self.max_history:
            self.hotspot_history.pop(0)
            self.time_history.pop(0)

    def predict_hotspot(self, dt: float) -> np.ndarray:
        """基于运动模型预测热点位置"""
        if len(self.hotspot_history) < 2:
            return self.hotspot_history[-1] if self.hotspot_history else np.array([50, 50])

        # 估计速度（使用最近几个点）
        n = min(5, len(self.hotspot_history))
        positions = np.array(self.hotspot_history[-n:])
        times = np.array(self.time_history[-n:])

        # 线性拟合估计速度
        if n >= 2:
            dt_hist = times[-1] - times[0]
            if dt_hist > 0.01:
                velocity = (positions[-1] - positions[0]) / dt_hist
            else:
                velocity = np.zeros(2)
        else:
            velocity = np.zeros(2)

        # 预测
        current_pos = self.hotspot_history[-1]
        predicted_pos = current_pos + velocity * dt

        # 边界约束
        predicted_pos[0] = np.clip(predicted_pos[0], self.domain[0] + 5, self.domain[1] - 5)
        predicted_pos[1] = np.clip(predicted_pos[1], self.domain[2] + 5, self.domain[3] - 5)

        return predicted_pos

    def compute_control(self, positions: np.ndarray,
                       current_hotspot: np.ndarray,
                       current_time: float,
                       prediction_horizon: float = 0.5) -> np.ndarray:
        """
        计算编队追踪控制
        """
        # 更新历史
        self.update_hotspot(current_hotspot, current_time)

        # 预测热点位置
        predicted_hotspot = self.predict_hotspot(prediction_horizon)

        # 计算���制
        return self.tracking_controller.compute_control(
            positions, current_hotspot, predicted_hotspot,
            current_time, prediction_horizon
        )