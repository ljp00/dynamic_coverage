"""
无人机智能体模块
"""
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class UAVState:
    """无人机状态"""
    position: np.ndarray
    velocity: np.ndarray
    heading: float = 0.0


class UAV:
    """无人机智能体"""

    def __init__(self, agent_id: int,
                 initial_position: np.ndarray,
                 max_velocity: float = 5.0,
                 max_acceleration: float = 2.0,
                 sensing_radius: float = 15.0):
        """
        Args:
            agent_id:  智能体ID
            initial_position: 初始位置
            max_velocity: 最大速度
            max_acceleration: 最大加速度
            sensing_radius: 感知半径
        """
        self.id = agent_id
        self.state = UAVState(
            position=np.array(initial_position, dtype=float),
            velocity=np.zeros(2)
        )
        self.v_max = max_velocity
        self.a_max = max_acceleration
        self.sensing_radius = sensing_radius

        # 轨迹历史
        self.trajectory: List[np.ndarray] = [self.state.position.copy()]

        # 感知数据缓存
        self.sensed_data: List[Tuple[np.ndarray, float, float]] = []  # (位置, 值, 时间)

    @property
    def position(self) -> np.ndarray:
        return self.state.position

    @property
    def velocity(self) -> np.ndarray:
        return self.state.velocity

    def update(self, control_input: np.ndarray, dt: float):
        """
        更新无人机状态（双积分器模型）

        Args:
            control_input: 加速度输入 (2,) 或速度输入
            dt: 时间步长
        """
        # 限制加速度
        acc = np.clip(control_input, -self.a_max, self.a_max)

        # 更新速度
        new_velocity = self.state.velocity + acc * dt

        # 限制速度
        speed = np.linalg.norm(new_velocity)
        if speed > self.v_max:
            new_velocity = new_velocity / speed * self.v_max

        # 更新位置
        self.state.position += 0.5 * (self.state.velocity + new_velocity) * dt
        self.state.velocity = new_velocity

        # 更新朝向
        if speed > 0.1:
            self.state.heading = np.arctan2(new_velocity[1], new_velocity[0])

        # 记录轨迹
        self.trajectory.append(self.state.position.copy())

    def set_velocity(self, velocity: np.ndarray, dt: float):
        """直接设置速度（一阶模型）"""
        # 限制速度
        speed = np.linalg.norm(velocity)
        if speed > self.v_max:
            velocity = velocity / speed * self.v_max

        # 更新位置
        self.state.position += velocity * dt
        self.state.velocity = velocity

        # 更新朝向
        if speed > 0.1:
            self.state.heading = np.arctan2(velocity[1], velocity[0])

        self.trajectory.append(self.state.position.copy())

    def sense(self, sensitivity_field, current_time: float) -> Tuple[np.ndarray, float]:
        """
        感知当前位置的敏感度值

        Returns:
            (位置, 敏感度值)
        """
        value = sensitivity_field.get_density(self.state.position)[0]
        self.sensed_data.append((self.state.position.copy(), value, current_time))
        return self.state.position.copy(), value

    def get_sensed_data(self, time_window: float = None,
                        current_time: float = None) -> np.ndarray:
        """获取感知数据用于GP训练"""
        if not self.sensed_data:
            return np.array([]), np.array([])

        data = self.sensed_data
        if time_window is not None and current_time is not None:
            data = [(p, v, t) for p, v, t in data if current_time - t <= time_window]

        if not data:
            return np.array([]), np.array([])

        X = np.array([[p[0], p[1], t] for p, v, t in data])
        y = np.array([v for p, v, t in data])
        return X, y


class UAVSwarm:
    """无人机集群"""

    def __init__(self, num_agents: int,
                 domain: Tuple[float, float, float, float],
                 **uav_kwargs):
        """
        Args:
            num_agents: 无人机数量
            domain: 工作区域
            **uav_kwargs: 传递给UAV的参数
        """
        self.num_agents = num_agents
        self.domain = domain

        # 初始化无人机（均匀分布）
        self.agents: List[UAV] = []
        positions = self._init_positions(num_agents, domain)
        for i, pos in enumerate(positions):
            self.agents.append(UAV(i, pos, **uav_kwargs))

    def _init_positions(self, n: int,
                        domain: Tuple[float, float, float, float]) -> np.ndarray:
        """初始化位置（网格分布+扰动）"""
        x_min, x_max, y_min, y_max = domain
        margin = 10

        # 计算网格
        aspect = (x_max - x_min) / (y_max - y_min)
        ny = int(np.sqrt(n / aspect))
        nx = int(np.ceil(n / ny))

        positions = []
        for i in range(nx):
            for j in range(ny):
                if len(positions) >= n:
                    break
                x = x_min + margin + (x_max - x_min - 2 * margin) * (i + 0.5) / nx
                y = y_min + margin + (y_max - y_min - 2 * margin) * (j + 0.5) / ny
                # 添加随机扰动
                x += np.random.uniform(-3, 3)
                y += np.random.uniform(-3, 3)
                positions.append([x, y])

        return np.array(positions[: n])

    def get_positions(self) -> np.ndarray:
        """获取所有无人机位置"""
        return np.array([agent.position for agent in self.agents])

    def get_velocities(self) -> np.ndarray:
        """获取所有无人机速度"""
        return np.array([agent.velocity for agent in self.agents])

    def update_all(self, controls: np.ndarray, dt: float, use_velocity: bool = True):
        """更新所有无人机"""
        for i, agent in enumerate(self.agents):
            if use_velocity:
                agent.set_velocity(controls[i], dt)
            else:
                agent.update(controls[i], dt)

    def sense_all(self, sensitivity_field, current_time: float):
        """所有无人机执行感知"""
        for agent in self.agents:
            agent.sense(sensitivity_field, current_time)

    def get_all_sensed_data(self, time_window: float = None,
                            current_time: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """汇总所有感知数据"""
        all_X, all_y = [], []
        for agent in self.agents:
            X, y = agent.get_sensed_data(time_window, current_time)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            return np.array([]), np.array([])

        return np.vstack(all_X), np.concatenate(all_y)