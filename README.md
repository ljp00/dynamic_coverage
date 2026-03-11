# 多无人机动态覆盖控制仿真框架

## 📋 概述

本框架实现了多无人机在**动态敏感度场**中的覆盖控制仿真，集成了以下核心功能：

- 🌡️ **动态敏感度场**:  时变高斯热点模型
- 🔮 **高斯过程(GP)预测**: 时空GP预测未来敏感度场
- 🎯 **覆盖控制**: 基于Voronoi分割的Lloyd算法
- 🛡️ **安全约束**:  CBF (Control Barrier Function) 与 MPC
- 📦 **任务分配**:  CBBA分布式拍卖算法

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 安装包（开发模式）

```bash
pip install -e .
```

### 运行仿真

```bash
# 基础仿真（5架无人机，30秒，启用CBF和GP预测）
python simulations/main_sim.py

# 对比实验（GP预测场可视化）
python simulations/comparison_experiment.py

# 基线 vs 预测对比
python simulations/baseline_vs_predictive.py

# 预测驱动仿真
python simulations/prediction_driven_sim.py

# 框架测试
python tests/test_framework.py
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | None |
| `--agents` | 无人机数量 | 5 |
| `--time` | 仿真时长(秒) | 60 |
| `--cbf` | 启用CBF安全约束 | True |
| `--mpc` | 启用MPC控制 | False |
| `--auction` | 启用任务分配 | False |
| `--no-viz` | 禁用可视化 | False |
| `--save-anim` | 保存动画 | False |

## 📁 项目结构

```
dynamic_coverage/
├── README.md
├── requirements.txt
├── setup.py                          # 包安装配置
├── .gitignore
│
├── config/
│   └── params.yaml                   # 配置参数
│
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── sensitivity_field.py      # 动态敏感度场（含 MotionType 枚举）
│   │   └── obstacles.py              # 障碍物环境
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── gp_predictor.py           # 高斯过程预测
│   ├── coverage/
│   │   ├── __init__.py
│   │   ├── voronoi.py                # Voronoi 分割
│   │   ├── lloyd_controller.py       # Lloyd 覆盖控制器
│   │   └── coverage_controllers.py   # 改进版控制器（含 Reactive + Predictive）
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── cbf.py                    # CBF 安全约束
│   │   └── mpc_controller.py         # MPC 控制
│   ├── allocation/
│   │   ├── __init__.py
│   │   └── auction.py                # 拍卖算法（CBBA）
│   ├── agents/
│   │   ├── __init__.py
│   │   └── uav.py                    # 无人机智能体
│   └── utils/
│       ├── __init__.py
│       └── visualization.py          # 可视化工具
│
├── simulations/
│   ├── __init__.py
│   ├── main_sim.py                   # 主仿真入口
│   ├── comparison_experiment.py      # GP 预测可视化对比实验
│   ├── baseline_vs_predictive.py     # 基线 vs 预测对比仿真
│   └── prediction_driven_sim.py      # 预测驱动仿真
│
├── tests/
│   ├── __init__.py
│   └── test_framework.py             # 框架测试脚本
│
└── output/
    └── .gitkeep
```

## 🔧 核心算法

### 1. 覆盖控制

基于加权Voronoi分割的Lloyd算法：

$$u_i = K(c_i^V - p_i)$$

其中 $c_i^V$ 是加权质心：

$$c_i^V = \frac{\int_{V_i} q \cdot \phi(q) dq}{\int_{V_i} \phi(q) dq}$$

### 2. GP预测

时空高斯过程核函数：

$$k(x_1, x_2) = \sigma_f^2 \exp\left(-\frac{||s_1-s_2||^2}{2l_s^2}\right) \exp\left(-\frac{|t_1-t_2|^2}{2l_t^2}\right)$$

### 3. CBF安全约束

碰撞避免障碍函数：

$$h_{ij}(p) = ||p_i - p_j||^2 - d_{safe}^2$$

CBF-QP安全滤波：

$$\min_u ||u - u_{nom}||^2 \quad \text{s.t. } \quad \nabla h \cdot u + \gamma h \geq 0$$

## 📊 仿真输出

- 实时可视化：敏感度场、无人机位置、Voronoi分割、GP预测
- 性能指标：覆盖代价曲线、改善率统计
- 可选输出：`coverage_simulation.mp4`, `simulation_results.png`

## 🔬 扩展方向

- [ ] 3D覆盖控制
- [ ] 异构无人机（不同感知/速度）
- [ ] 通信约束建模
- [ ] 深度强化学习覆盖策略
- [ ] ROS/Gazebo集成

## 📚 参考文献

1. Cortes, J., et al. "Coverage control for mobile sensing networks." IEEE TRO, 2004. 
2. Schwager, M., et al. "Decentralized, adaptive coverage control." IJRR, 2009.
3. Ames, A., et al. "Control barrier functions." IEEE TAC, 2017.