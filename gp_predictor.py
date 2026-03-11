"""
高斯过程预测模块
用于预测未来敏感度场的时空演化
"""
import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cho_factor


class SpatioTemporalGP:
    """时空高斯过程预测器"""

    def __init__(self, length_scale_space: float = 10.0,
                 length_scale_time: float = 5.0,
                 signal_variance: float = 1.0,
                 noise_variance: float = 0.1):
        """
        Args:
            length_scale_space: 空间长度尺度
            length_scale_time: 时间长度尺度
            signal_variance: 信号方差
            noise_variance: 噪声方差
        """
        self.l_s = length_scale_space
        self.l_t = length_scale_time
        self.sigma_f = signal_variance
        self.sigma_n = noise_variance

        # 训练数据
        self.X_train = None  # shape (N, 3): [x, y, t]
        self.y_train = None  # shape (N,)
        self.K_inv = None
        self.alpha = None

    def kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        时空RBF核函数
        K(x1, x2) = σ_f² * exp(-||x1-x2||²/(2*l_s²)) * exp(-|t1-t2|²/(2*l_t²))
        """
        # 分离空间和时间维度
        space1, time1 = X1[:, :2], X1[:, 2:3]
        space2, time2 = X2[:, :2], X2[:, 2:3]

        # 空间距离
        dist_space = cdist(space1, space2, 'sqeuclidean')
        # 时间距离
        dist_time = cdist(time1, time2, 'sqeuclidean')

        K = self.sigma_f ** 2 * np.exp(-dist_space / (2 * self.l_s ** 2)) * \
            np.exp(-dist_time / (2 * self.l_t ** 2))
        return K

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练高斯过程
        Args:
            X: 训练输入 shape (N, 3) [x, y, t]
            y: 训练输出 shape (N,)
        """
        self.X_train = X
        self.y_train = y

        # 计算核矩阵
        K = self.kernel(X, X)
        K += self.sigma_n ** 2 * np.eye(len(X))  # 添加噪声

        # Cholesky分解用于稳定求逆
        try:
            self.L = cho_factor(K)
            self.alpha = cho_solve(self.L, y)
        except np.linalg.LinAlgError:
            # 添加抖动以提高数值稳定性
            K += 1e-6 * np.eye(len(X))
            self.L = cho_factor(K)
            self.alpha = cho_solve(self.L, y)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测新位置的敏感度
        Args:
            X_test: 测试输入 shape (M, 3) [x, y, t]
        Returns:
            mean:  预测均值 shape (M,)
            var: 预测方差 shape (M,)
        """
        if self.X_train is None:
            raise ValueError("模型尚未训练，请先调用fit()")

        # 计算测试点与训练点的核
        K_star = self.kernel(X_test, self.X_train)

        # 预测均值
        mean = K_star @ self.alpha

        # 预测方差
        v = cho_solve(self.L, K_star.T)
        K_star_star = self.kernel(X_test, X_test)
        var = np.diag(K_star_star) - np.sum(K_star * v.T, axis=1)
        var = np.maximum(var, 1e-10)  # 确保非负

        return mean, var

    def update_online(self, x_new: np.ndarray, y_new: float):
        """
        在线更新GP（简化版，增量添加数据点）
        """
        if self.X_train is None:
            self.X_train = x_new.reshape(1, -1)
            self.y_train = np.array([y_new])
        else:
            self.X_train = np.vstack([self.X_train, x_new])
            self.y_train = np.append(self.y_train, y_new)

            # 限制训练数据大小以保持计算效率
            max_points = 500
            if len(self.X_train) > max_points:
                # 保留最近的数据点
                indices = np.argsort(self.X_train[:, 2])[-max_points:]
                self.X_train = self.X_train[indices]
                self.y_train = self.y_train[indices]

        self.fit(self.X_train, self.y_train)


class SparseGP(SpatioTemporalGP):
    """稀疏高斯过程（用于大规模数据）"""

    def __init__(self, num_inducing: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.num_inducing = num_inducing
        self.Z = None  # 诱导点

    def select_inducing_points(self, X: np.ndarray):
        """使用K-means选择诱导点"""
        from scipy.cluster.vq import kmeans2
        if len(X) <= self.num_inducing:
            self.Z = X.copy()
        else:
            self.Z, _ = kmeans2(X, self.num_inducing, minit='points')

    def fit(self, X: np.ndarray, y: np.ndarray):
        """稀疏GP训练（FITC近似）"""
        self.X_train = X
        self.y_train = y

        # 选择诱导点
        self.select_inducing_points(X)

        # 计算必要的核矩阵
        Kuu = self.kernel(self.Z, self.Z) + 1e-6 * np.eye(len(self.Z))
        Kuf = self.kernel(self.Z, X)

        # FITC近似
        self.Luu = cho_factor(Kuu)
        V = cho_solve(self.Luu, Kuf)

        # 对角近似
        Qff_diag = np.sum(V * Kuf, axis=0)
        Kff_diag = self.sigma_f ** 2 * np.ones(len(X))
        Lambda_diag = Kff_diag - Qff_diag + self.sigma_n ** 2

        # 计算预测所需的量
        Lambda_inv_y = y / Lambda_diag
        self.Sigma = Kuu + Kuf @ (Kuf.T / Lambda_diag[:, None])
        self.L_sigma = cho_factor(self.Sigma)
        self.alpha_sparse = cho_solve(self.L_sigma, Kuf @ Lambda_inv_y)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """稀疏GP预测"""
        Kus = self.kernel(self.Z, X_test)

        # 预测均值
        mean = Kus.T @ self.alpha_sparse

        # 预测方差（简化计算）
        v = cho_solve(self.Luu, Kus)
        var = self.sigma_f ** 2 - np.sum(v * Kus, axis=0)
        var = np.maximum(var, 1e-10)

        return mean, var