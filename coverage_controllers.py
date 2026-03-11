"""
覆盖控制器（改进版）
修复：GP预测质量评估 + 自适应权重 + 更好的学习策略
"""
import numpy as np
from typing import Tuple, Dict, Optional, List


class LocalSensor:
    """局部感知传感器"""

    def __init__(self, sensing_radius: float = 15.0, num_samples: int = 12):
        self.sensing_radius = sensing_radius
        self.num_samples = num_samples

    def sense(self, position: np.ndarray, field_func, current_time: float = None) -> Dict:
        """在当前位置进行局部感知"""
        current_value = field_func(position.reshape(1, -1))[0]

        samples = []
        for i in range(self.num_samples):
            angle = 2 * np.pi * i / self.num_samples
            for r_ratio in [0.3, 0.6, 1.0]:
                r = self.sensing_radius * r_ratio
                sample_pos = position + r * np.array([np.cos(angle), np.sin(angle)])
                sample_value = field_func(sample_pos.reshape(1, -1))[0]
                samples.append({
                    'position': sample_pos.copy(),
                    'value': sample_value
                })

        return {
            'position': position.copy(),
            'value': current_value,
            'time': current_time,
            'samples': samples
        }


class ReactiveLocalController:
    """
    基线算法：反应式局部感知控制器
    """

    def __init__(self, domain: Tuple = (0, 100, 0, 100),
                 max_velocity: float = 5.0,
                 sensing_radius: float = 15.0,
                 gradient_gain: float = 2.0,
                 local_max_gain: float = 1.5,
                 separation_gain: float = 2.0,
                 separation_radius: float = 10.0):

        self.domain = domain
        self.max_velocity = max_velocity
        self.sensing_radius = sensing_radius
        self.gradient_gain = gradient_gain
        self.local_max_gain = local_max_gain
        self.separation_gain = separation_gain
        self.separation_radius = separation_radius
        self.sensors = {}

    def initialize_agent(self, agent_id: int):
        self.sensors[agent_id] = LocalSensor(
            sensing_radius=self.sensing_radius,
            num_samples=12
        )

    def compute_control(self,
                        agent_id: int,
                        position: np.ndarray,
                        field_func,
                        current_time: float,
                        other_positions: np.ndarray = None) -> Tuple[np.ndarray, Dict]:

        if agent_id not in self.sensors:
            self.initialize_agent(agent_id)

        sensing_data = self.sensors[agent_id].sense(position, field_func, current_time)
        gradient = self._compute_local_gradient(position, sensing_data)
        local_max_dir = self._find_local_maximum_direction(position, sensing_data)

        velocity = self.gradient_gain * gradient + self.local_max_gain * local_max_dir

        if other_positions is not None and len(other_positions) > 0:
            velocity += self._compute_separation(position, other_positions)

        velocity += self._compute_boundary_force(position)
        velocity = self._limit_velocity(velocity)

        return velocity, {'mode': 'reactive', 'gradient': gradient}

    def _compute_local_gradient(self, position: np.ndarray, sensing_data: Dict) -> np.ndarray:
        gradient = np.zeros(2)
        center_value = sensing_data['value']

        for sample in sensing_data['samples']:
            direction = sample['position'] - position
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                value_diff = sample['value'] - center_value
                gradient += value_diff * direction / (dist ** 2 + 1e-6)

        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm
        return gradient

    def _find_local_maximum_direction(self, position: np.ndarray, sensing_data: Dict) -> np.ndarray:
        best_value = sensing_data['value']
        best_direction = np.zeros(2)

        for sample in sensing_data['samples']:
            if sample['value'] > best_value:
                best_value = sample['value']
                direction = sample['position'] - position
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    best_direction = direction / dist
        return best_direction

    def _compute_separation(self, position: np.ndarray, other_positions: np.ndarray) -> np.ndarray:
        separation = np.zeros(2)
        for other_pos in other_positions:
            diff = position - other_pos
            dist = np.linalg.norm(diff)
            if 0.1 < dist < self.separation_radius:
                force = self.separation_gain * (1 - dist / self.separation_radius)
                separation += force * diff / dist
        return separation

    def _compute_boundary_force(self, position: np.ndarray) -> np.ndarray:
        force = np.zeros(2)
        margin, strength = 10.0, 3.0

        if position[0] < self.domain[0] + margin:
            force[0] += strength * (self.domain[0] + margin - position[0])
        if position[0] > self.domain[1] - margin:
            force[0] -= strength * (position[0] - (self.domain[1] - margin))
        if position[1] < self.domain[2] + margin:
            force[1] += strength * (self.domain[2] + margin - position[1])
        if position[1] > self.domain[3] - margin:
            force[1] -= strength * (position[1] - (self.domain[3] - margin))
        return force

    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed > self.max_velocity:
            velocity = velocity / speed * self.max_velocity
        return velocity


class ImprovedGPPredictor:
    """
    改进的GP预测器
    增加：预测质量评估、更好的训练策略
    """

    def __init__(self, domain: Tuple = (0, 100, 0, 100), resolution: int = 20):
        self.domain = domain
        self.resolution = resolution

        # 训练数据
        self.X_train = []
        self.y_train = []
        self.max_data = 800  # 增加数据容量

        # GP参数
        self.length_scale_space = 15.0  # 减小空间尺度，更敏感
        self.length_scale_time = 3.0    # 减小时间尺度，更敏感
        self.noise_var = 0.05           # 减小噪声

        # 训练状态
        self.is_trained = False
        self.K_inv = None
        self.alpha = None

        # 预测质量评估
        self.prediction_history = []
        self.actual_history = []
        self.prediction_error_estimate = 50.0  # 初始假设误差很大

    def add_observation(self, position: np.ndarray, value: float, time: float):
        """添加观测数据"""
        self.X_train.append([position[0], position[1], time])
        self.y_train.append(value)

        if len(self.X_train) > self.max_data:
            self.X_train.pop(0)
            self.y_train.pop(0)

    def add_sensing_data(self, sensing_data: Dict):
        """添加感知数据"""
        if sensing_data['time'] is None:
            return

        self.add_observation(
            sensing_data['position'],
            sensing_data['value'],
            sensing_data['time']
        )

        for sample in sensing_data['samples']:
            self.add_observation(
                sample['position'],
                sample['value'],
                sensing_data['time']
            )

    def train(self) -> bool:
        """训练GP模型"""
        if len(self.X_train) < 50:  # 增加最小数据要求
            return False

        X = np.array(self.X_train)
        y = np.array(self.y_train)

        K = self._compute_kernel(X, X)
        K += self.noise_var * np.eye(len(X))

        try:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(len(K)))
            self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self.K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(X))))
            self.is_trained = True
            return True
        except:
            self.is_trained = False
            return False

    def predict(self, positions: np.ndarray, time: float) -> Tuple[np.ndarray, np.ndarray]:
        """预测指定位置和时间的敏感度"""
        if not self.is_trained:
            return np.zeros(len(positions)), np.ones(len(positions)) * 100

        X_train = np.array(self.X_train)
        X_test = np.column_stack([positions, np.full(len(positions), time)])

        K_star = self._compute_kernel(X_test, X_train)
        mean = K_star @ self.alpha

        K_star_star = self._compute_kernel_diag(X_test)
        var = K_star_star - np.sum((K_star @ self.K_inv) * K_star, axis=1)
        var = np.maximum(var, 1e-6)

        return mean, var

    def predict_field(self, time: float) -> Tuple[np.ndarray, float]:
        """预测完整敏感度场，返回场和平均方差"""
        x_grid = np.linspace(self.domain[0], self.domain[1], self.resolution)
        y_grid = np.linspace(self.domain[2], self.domain[3], self.resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.column_stack([X.ravel(), Y.ravel()])

        mean, var = self.predict(positions, time)
        avg_var = np.mean(var)

        return np.maximum(mean.reshape(self.resolution, self.resolution), 0.1), avg_var

    def predict_hotspot(self, time: float) -> Tuple[Optional[np.ndarray], float]:
        """预测热点位置，返回位置和置信度"""
        if not self.is_trained:
            return None, 0.0

        field, avg_var = self.predict_field(time)
        max_idx = np.argmax(field)
        row, col = np.unravel_index(max_idx, field.shape)

        x = self.domain[0] + (self.domain[1] - self.domain[0]) * col / (self.resolution - 1)
        y = self.domain[2] + (self.domain[3] - self.domain[2]) * row / (self.resolution - 1)

        # 置信度基于方差和历史误差
        confidence = 1.0 / (1.0 + avg_var + self.prediction_error_estimate / 50.0)
        confidence = np.clip(confidence, 0.1, 0.9)

        return np.array([x, y]), confidence

    def update_prediction_quality(self, predicted_hotspot: np.ndarray,
                                   actual_local_max: np.ndarray):
        """更新预测质量估计"""
        if predicted_hotspot is None or actual_local_max is None:
            return

        error = np.linalg.norm(predicted_hotspot - actual_local_max)

        # 指数移动平均更新误差估计
        alpha = 0.2
        self.prediction_error_estimate = (1 - alpha) * self.prediction_error_estimate + alpha * error

    def get_prediction_confidence(self) -> float:
        """获取当前预测置信度"""
        if not self.is_trained:
            return 0.0

        # 基于数据量和误差估计
        data_confidence = min(1.0, len(self.X_train) / 200.0)
        error_confidence = 1.0 / (1.0 + self.prediction_error_estimate / 20.0)

        return data_confidence * error_confidence

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        spatial_dist_sq = (
            (X1[:, 0:1] - X2[:, 0:1].T) ** 2 +
            (X1[:, 1:2] - X2[:, 1:2].T) ** 2
        )
        time_dist_sq = (X1[:, 2:3] - X2[:, 2:3].T) ** 2

        K = np.exp(-spatial_dist_sq / (2 * self.length_scale_space ** 2) -
                   time_dist_sq / (2 * self.length_scale_time ** 2))
        return K

    def _compute_kernel_diag(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X))


class PredictiveGPController:
    """
    改进的GP预测控制器

    关键改进：
    1. 自适应预测权重（基于预测置信度）
    2. 预测质量在线评估
    3. 更长的学习阶段
    4. 预测失败时平滑退化到反应式
    """

    def __init__(self, domain: Tuple = (0, 100, 0, 100),
                 max_velocity: float = 5.0,
                 sensing_radius: float = 15.0,
                 prediction_horizon: float = 0.5,  # 减小预测步长
                 learning_duration: float = 15.0,   # 增加学习时间
                 max_prediction_weight: float = 0.5,  # 降低最大预测权重
                 formation_radius: float = 12.0,
                 position_gain: float = 1.2,
                 velocity_gain: float = 0.5,
                 separation_gain: float = 2.0,
                 separation_radius: float = 10.0):

        self.domain = domain
        self.max_velocity = max_velocity
        self.sensing_radius = sensing_radius
        self.prediction_horizon = prediction_horizon
        self.learning_duration = learning_duration
        self.max_prediction_weight = max_prediction_weight
        self.formation_radius = formation_radius
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.separation_gain = separation_gain
        self.separation_radius = separation_radius

        self.sensors = {}
        self.gp_predictor = ImprovedGPPredictor(domain, resolution=25)

        # 热点估计
        self.estimated_hotspot_history = []
        self.time_history = []
        self.estimated_velocity = np.zeros(2)

        # 训练控制
        self.gp_update_interval = 0.5  # 更频繁更新
        self.last_gp_update = -np.inf

        # 局部最大值追踪（用于评估预测质量）
        self.local_max_history = []

    def initialize_agent(self, agent_id: int):
        self.sensors[agent_id] = LocalSensor(
            sensing_radius=self.sensing_radius,
            num_samples=16  # 增加采样点
        )

    def compute_control(self,
                        agent_id: int,
                        position: np.ndarray,
                        field_func,
                        current_time: float,
                        other_positions: np.ndarray = None,
                        n_agents: int = 3) -> Tuple[np.ndarray, Dict]:

        if agent_id not in self.sensors:
            self.initialize_agent(agent_id)

        # 1. 局部感知
        sensing_data = self.sensors[agent_id].sense(position, field_func, current_time)

        # 2. 添加到GP训练数据
        self.gp_predictor.add_sensing_data(sensing_data)

        # 3. 找到局部感知的最大值位置（用于评估预测质量）
        local_max_pos = self._find_local_max_position(position, sensing_data)

        # 判断阶段
        if current_time < self.learning_duration:
            # 学习阶段
            velocity, info = self._learning_phase_control(
                agent_id, position, sensing_data, other_positions, n_agents
            )
            info['mode'] = 'learning'

            # 定期训练GP
            if current_time - self.last_gp_update > self.gp_update_interval:
                self.gp_predictor.train()
                self.last_gp_update = current_time
        else:
            # 部署阶段
            if not self.gp_predictor.is_trained:
                self.gp_predictor.train()

            velocity, info = self._deployment_phase_control(
                agent_id, position, sensing_data, current_time,
                other_positions, n_agents, local_max_pos
            )

        return velocity, info

    def _learning_phase_control(self,
                                 agent_id: int,
                                 position: np.ndarray,
                                 sensing_data: Dict,
                                 other_positions: np.ndarray,
                                 n_agents: int) -> Tuple[np.ndarray, Dict]:
        """学习阶段：探索 + 局部感知"""

        gradient = self._compute_gradient(position, sensing_data)
        local_max_dir = self._find_local_max_direction(position, sensing_data)

        # 探索分量：鼓励分散覆盖
        exploration = self._compute_exploration(agent_id, position, other_positions, n_agents)

        # 学习阶段更强调探索
        velocity = 1.0 * gradient + 0.8 * local_max_dir + 1.0 * exploration

        if other_positions is not None:
            velocity += self._compute_separation(position, other_positions)
        velocity += self._compute_boundary_force(position)
        velocity = self._limit_velocity(velocity)

        return velocity, {'gradient': gradient, 'exploration': exploration}

    def _deployment_phase_control(self,
                                   agent_id: int,
                                   position: np.ndarray,
                                   sensing_data: Dict,
                                   current_time: float,
                                   other_positions: np.ndarray,
                                   n_agents: int,
                                   local_max_pos: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """部署阶段：预测 + 反应式混合"""

        # 预测热点
        future_time = current_time + self.prediction_horizon
        predicted_hotspot, pred_confidence = self.gp_predictor.predict_hotspot(future_time)

        # 更新预测质量评估
        if predicted_hotspot is not None and local_max_pos is not None:
            self.gp_predictor.update_prediction_quality(predicted_hotspot, local_max_pos)

        # 计算反应式控制
        gradient = self._compute_gradient(position, sensing_data)
        local_max_dir = self._find_local_max_direction(position, sensing_data)
        vel_reactive = 2.0 * gradient + 1.5 * local_max_dir

        # 计算预测控制
        if predicted_hotspot is not None and pred_confidence > 0.2:
            # 更新热点速度估计
            self._update_hotspot_velocity(predicted_hotspot, current_time)

            # 编队目标
            target = self._compute_formation_target(agent_id, predicted_hotspot, n_agents)
            vel_predictive = self._compute_predictive_control(position, target)

            # 自适应权重：置信度越高，预测权重越大
            alpha = min(self.max_prediction_weight, pred_confidence * self.max_prediction_weight)

            velocity = alpha * vel_predictive + (1 - alpha) * vel_reactive
            mode = 'predictive'
        else:
            # 预测不可靠，使用纯反应式
            velocity = vel_reactive
            alpha = 0.0
            mode = 'reactive_fallback'

        # 分离和边界
        if other_positions is not None:
            velocity += self._compute_separation(position, other_positions)
        velocity += self._compute_boundary_force(position)
        velocity = self._limit_velocity(velocity)

        info = {
            'mode': mode,
            'prediction_confidence': pred_confidence if predicted_hotspot is not None else 0,
            'prediction_weight': alpha,
            'predicted_hotspot': predicted_hotspot
        }

        return velocity, info

    def _compute_gradient(self, position: np.ndarray, sensing_data: Dict) -> np.ndarray:
        gradient = np.zeros(2)
        center_value = sensing_data['value']

        for sample in sensing_data['samples']:
            direction = sample['position'] - position
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                value_diff = sample['value'] - center_value
                gradient += value_diff * direction / (dist ** 2 + 1e-6)

        norm = np.linalg.norm(gradient)
        if norm > 1e-6:
            gradient = gradient / norm
        return gradient

    def _find_local_max_direction(self, position: np.ndarray, sensing_data: Dict) -> np.ndarray:
        best_value = sensing_data['value']
        best_direction = np.zeros(2)

        for sample in sensing_data['samples']:
            if sample['value'] > best_value:
                best_value = sample['value']
                direction = sample['position'] - position
                dist = np.linalg.norm(direction)
                if dist > 0.1:
                    best_direction = direction / dist
        return best_direction

    def _find_local_max_position(self, position: np.ndarray, sensing_data: Dict) -> np.ndarray:
        """找到感知范围内的最大值位置"""
        best_value = sensing_data['value']
        best_pos = position.copy()

        for sample in sensing_data['samples']:
            if sample['value'] > best_value:
                best_value = sample['value']
                best_pos = sample['position'].copy()
        return best_pos

    def _compute_exploration(self, agent_id: int, position: np.ndarray,
                             other_positions: np.ndarray, n_agents: int) -> np.ndarray:
        """探索控制：智能体分散到不同区域"""
        # 目标区域（根据智能体ID分配）
        angle = 2 * np.pi * agent_id / n_agents
        target_region = np.array([50, 50]) + 30 * np.array([np.cos(angle), np.sin(angle)])

        direction = target_region - position
        dist = np.linalg.norm(direction)
        if dist > 1.0:
            return 0.5 * direction / dist
        return np.zeros(2)

    def _update_hotspot_velocity(self, hotspot: np.ndarray, current_time: float):
        self.estimated_hotspot_history.append(hotspot.copy())
        self.time_history.append(current_time)

        if len(self.estimated_hotspot_history) > 20:
            self.estimated_hotspot_history.pop(0)
            self.time_history.pop(0)

        if len(self.estimated_hotspot_history) >= 2:
            dt = self.time_history[-1] - self.time_history[-2]
            if dt > 0.01:
                raw_vel = (self.estimated_hotspot_history[-1] - self.estimated_hotspot_history[-2]) / dt
                alpha = 0.3
                self.estimated_velocity = alpha * raw_vel + (1 - alpha) * self.estimated_velocity

    def _compute_formation_target(self, agent_id: int, hotspot: np.ndarray, n_agents: int) -> np.ndarray:
        angle = 2 * np.pi * agent_id / n_agents
        offset = self.formation_radius * np.array([np.cos(angle), np.sin(angle)])
        return hotspot + offset

    def _compute_predictive_control(self, position: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos_error = target - position
        velocity = self.position_gain * pos_error + self.velocity_gain * self.estimated_velocity
        return velocity

    def _compute_separation(self, position: np.ndarray, other_positions: np.ndarray) -> np.ndarray:
        separation = np.zeros(2)
        for other_pos in other_positions:
            diff = position - other_pos
            dist = np.linalg.norm(diff)
            if 0.1 < dist < self.separation_radius:
                force = self.separation_gain * (1 - dist / self.separation_radius)
                separation += force * diff / dist
        return separation

    def _compute_boundary_force(self, position: np.ndarray) -> np.ndarray:
        force = np.zeros(2)
        margin, strength = 10.0, 3.0

        if position[0] < self.domain[0] + margin:
            force[0] += strength * (self.domain[0] + margin - position[0])
        if position[0] > self.domain[1] - margin:
            force[0] -= strength * (position[0] - (self.domain[1] - margin))
        if position[1] < self.domain[2] + margin:
            force[1] += strength * (self.domain[2] + margin - position[1])
        if position[1] > self.domain[3] - margin:
            force[1] -= strength * (position[1] - (self.domain[3] - margin))
        return force

    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        speed = np.linalg.norm(velocity)
        if speed > self.max_velocity:
            velocity = velocity / speed * self.max_velocity
        return velocity