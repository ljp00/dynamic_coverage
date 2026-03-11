"""
拍卖算法任务分配模块
用于分配监控区域或特定目标
"""
import numpy as np
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, field


@dataclass
class Task:
    """任务定义"""
    id: int
    position: np.ndarray  # 任务位置
    priority: float  # 优先级/价值
    deadline: float = np.inf  # 截止时间
    required_time: float = 1.0  # 完成所需时间


@dataclass
class Bid:
    """投标"""
    agent_id: int
    task_id: int
    value: float  # 投标价值


class SequentialAuction:
    """序贯拍卖算法"""

    def __init__(self, num_agents: int):
        self.num_agents = num_agents

    def compute_task_value(self, agent_pos: np.ndarray,
                           task: Task,
                           agent_velocity: float = 5.0) -> float:
        """
        计算任务价值（考虑距离和优先级）
        """
        distance = np.linalg.norm(agent_pos - task.position)
        travel_time = distance / agent_velocity

        # 价值 = 优先级 / (旅行时间 + 执行时间)
        value = task.priority / (travel_time + task.required_time + 1e-6)

        # 考虑截止时间
        if travel_time > task.deadline:
            value = 0

        return value

    def allocate(self, agent_positions: np.ndarray,
                 tasks: List[Task]) -> Dict[int, List[int]]:
        """
        执行序贯拍卖分配

        Args:
            agent_positions: 智能体位置 shape (N, 2)
            tasks: 任务列表
        Returns:
            allocation: {agent_id: [task_ids]}
        """
        num_tasks = len(tasks)
        allocation = {i: [] for i in range(self.num_agents)}
        assigned_tasks: Set[int] = set()

        # 依次分配每个任务
        for _ in range(num_tasks):
            best_bid = None
            best_value = -np.inf

            # 收集所有智能体对未分配任务的投标
            for agent_id in range(self.num_agents):
                for task in tasks:
                    if task.id in assigned_tasks:
                        continue

                    # 计算考虑已分配任务后的边际价值
                    current_pos = agent_positions[agent_id].copy()

                    # 如果已有分配任务，从最后一个任务位置出发
                    if allocation[agent_id]:
                        last_task_id = allocation[agent_id][-1]
                        last_task = next(t for t in tasks if t.id == last_task_id)
                        current_pos = last_task.position

                    value = self.compute_task_value(current_pos, task)

                    if value > best_value:
                        best_value = value
                        best_bid = Bid(agent_id, task.id, value)

            # 分配最高价值的投标
            if best_bid is not None and best_value > 0:
                allocation[best_bid.agent_id].append(best_bid.task_id)
                assigned_tasks.add(best_bid.task_id)

        return allocation


class CBBA:
    """
    Consensus-Based Bundle Algorithm (CBBA)
    分布式一致性任务分配算法
    """

    def __init__(self, num_agents: int, max_bundle_size: int = 3):
        """
        Args:
            num_agents: 智能体数量
            max_bundle_size:  每个智能体最大任务数
        """
        self.num_agents = num_agents
        self.max_bundle = max_bundle_size

    def allocate(self, agent_positions: np.ndarray,
                 tasks: List[Task],
                 max_iterations: int = 50,
                 communication_graph: np.ndarray = None) -> Dict[int, List[int]]:
        """
        执行CBBA分配

        Args:
            agent_positions:  智能体位置
            tasks: 任务列表
            max_iterations: 最大迭代次数
            communication_graph: 通信图邻接矩阵 (默认全连接)
        Returns:
            allocation: {agent_id: [task_ids]}
        """
        num_tasks = len(tasks)

        if communication_graph is None:
            communication_graph = np.ones((self.num_agents, self.num_agents))
            np.fill_diagonal(communication_graph, 0)

        # 初始化
        # 任务价值矩阵 y[i,j] = agent i 对 task j 的最高已知投标
        y = np.zeros((self.num_agents, num_tasks))
        # 中标者矩阵 z[i,j] = agent i 认为的 task j 的中标者
        z = -np.ones((self.num_agents, num_tasks), dtype=int)
        # 时间戳 s[i,j] = agent i 关于 task j 信息的更新时间
        s = np.zeros((self.num_agents, num_tasks))

        # Bundle和Path
        bundles = [[] for _ in range(self.num_agents)]  # 任务bundle
        paths = [[] for _ in range(self.num_agents)]  # 执行顺序

        for iteration in range(max_iterations):
            # Phase 1: Bundle Building (每个智能体独立执行)
            for i in range(self.num_agents):
                while len(bundles[i]) < self.max_bundle:
                    # 找到最佳未分配任务
                    best_task = -1
                    best_value = -np.inf
                    best_insert_pos = 0

                    for j, task in enumerate(tasks):
                        # 跳过已在bundle中的任务
                        if j in bundles[i]:
                            continue

                        # 计算将任务插入path各位置的边际价值
                        for insert_pos in range(len(paths[i]) + 1):
                            value = self._compute_marginal_value(
                                agent_positions[i], tasks, paths[i], j, insert_pos
                            )

                            # 只有当价值高于当前最高投标时才投标
                            if value > y[i, j] and value > best_value:
                                best_value = value
                                best_task = j
                                best_insert_pos = insert_pos

                    if best_task >= 0:
                        # 添加到bundle和path
                        bundles[i].append(best_task)
                        paths[i].insert(best_insert_pos, best_task)
                        y[i, best_task] = best_value
                        z[i, best_task] = i
                        s[i, best_task] = iteration
                    else:
                        break

            # Phase 2: Consensus (智能体间通信)
            y_new = y.copy()
            z_new = z.copy()
            s_new = s.copy()

            for i in range(self.num_agents):
                for k in range(self.num_agents):
                    if communication_graph[i, k] == 0:
                        continue

                    # 从邻居k接收信息并更新
                    for j in range(num_tasks):
                        # 使用更新规则
                        if s[k, j] > s_new[i, j]:
                            if y[k, j] > y_new[i, j]:
                                y_new[i, j] = y[k, j]
                                z_new[i, j] = z[k, j]
                                s_new[i, j] = s[k, j]

                                # 如果更新导致冲突，从bundle中移除
                                if z_new[i, j] != i and j in bundles[i]:
                                    idx = bundles[i].index(j)
                                    # 移除该任务及其后所有任务
                                    removed = bundles[i][idx:]
                                    bundles[i] = bundles[i][:idx]
                                    paths[i] = [t for t in paths[i] if t not in removed]
                                    # 重置这些任务的投标
                                    for t in removed:
                                        y_new[i, t] = 0
                                        z_new[i, t] = -1

            y, z, s = y_new, z_new, s_new

        # 构建最终分配结果
        allocation = {i: bundles[i] for i in range(self.num_agents)}
        return allocation

    def _compute_marginal_value(self, agent_pos: np.ndarray,
                                tasks: List[Task],
                                current_path: List[int],
                                new_task_id: int,
                                insert_pos: int,
                                agent_velocity: float = 5.0) -> float:
        """计算将新任务插入path指定位置的边际价值"""
        new_task = tasks[new_task_id]

        # 计算插入前的路径代价
        if not current_path:
            # 空path，直接计算到任务的价值
            distance = np.linalg.norm(agent_pos - new_task.position)
            travel_time = distance / agent_velocity
            return new_task.priority / (travel_time + new_task.required_time + 0.1)

        # 计算插入后的额外代价
        # 获取插入位置前后的位置
        if insert_pos == 0:
            prev_pos = agent_pos
        else:
            prev_task = tasks[current_path[insert_pos - 1]]
            prev_pos = prev_task.position

        if insert_pos == len(current_path):
            next_pos = None
        else:
            next_task = tasks[current_path[insert_pos]]
            next_pos = next_task.position

        # 计算额外旅行距离
        dist_to_new = np.linalg.norm(prev_pos - new_task.position)
        if next_pos is not None:
            dist_new_to_next = np.linalg.norm(new_task.position - next_pos)
            dist_direct = np.linalg.norm(prev_pos - next_pos)
            extra_dist = dist_to_new + dist_new_to_next - dist_direct
        else:
            extra_dist = dist_to_new

        extra_time = extra_dist / agent_velocity + new_task.required_time

        # 边际价值 = 任务价值 - 额外代价
        marginal_value = new_task.priority - extra_time * 0.5

        return max(0, marginal_value)