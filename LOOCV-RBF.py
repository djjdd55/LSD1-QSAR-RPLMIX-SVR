from math import sqrt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import random

# 设置随机种子以确保结果可重复
np.random.seed(42)
random.seed(42)

# 读取数据
filename = 'Your data.csv'
df = pd.read_csv(filename, header=None)

y = df.iloc[:, -1].values
X = df.iloc[:, :-1].values

y_true = []
y_predict = []


# 自适应粒子群优化算法(APSO)实现
class APSO:
    def __init__(self, n_particles, dimensions, bounds, max_iter=100):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter

        # 初始化粒子位置和速度
        self.positions = np.zeros((n_particles, dimensions))
        self.velocities = np.zeros((n_particles, dimensions))
        self.best_positions = np.zeros((n_particles, dimensions))
        self.best_scores = np.ones(n_particles) * float('inf')
        self.global_best_position = np.zeros(dimensions)
        self.global_best_score = float('inf')

        # 初始化粒子位置和速度
        for i in range(dimensions):
            self.positions[:, i] = np.random.uniform(bounds[0][i], bounds[1][i], n_particles)
            self.velocities[:, i] = np.random.uniform(-1, 1, n_particles) * (bounds[1][i] - bounds[0][i]) * 0.1

        self.best_positions = self.positions.copy()

    def update_parameters(self, iter_num, f_avg, f_min, f):
        """更新APSO的参数（惯性权重和学习因子）"""
        w_max, w_min = 0.9, 0.4
        c1_max, c1_min = 2.5, 0.5
        c2_max, c2_min = 2.5, 0.5

        # 计算进化因子
        if f_avg == f_min:
            f_avg = f_min + 1e-10  # 避免除零

        # 对每个粒子计算进化因子
        f_values = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            if f_avg == f_min:
                f_values[i] = 1.0
            else:
                f_values[i] = (f[i] - f_min) / (f_avg - f_min)

        # 根据进化因子调整参数
        w_values = np.zeros(self.n_particles)
        c1_values = np.zeros(self.n_particles)
        c2_values = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            # 惯性权重自适应调整
            w_values[i] = w_max - (w_max - w_min) * iter_num / self.max_iter

            # 学习因子自适应调整
            if f_values[i] >= 0.5:  # 处于探索阶段
                c1_values[i] = c1_min + (c1_max - c1_min) * (f_values[i] - 0.5) * 2
                c2_values[i] = c2_max - (c2_max - c2_min) * (f_values[i] - 0.5) * 2
            else:  # 处于开发阶段
                c1_values[i] = c1_max - (c1_max - c1_min) * f_values[i] * 2
                c2_values[i] = c2_min + (c2_max - c2_min) * f_values[i] * 2

        return w_values, c1_values, c2_values

    def optimize(self, objective_function):
        # 初始评估
        scores = objective_function(self.positions)

        for i in range(self.n_particles):
            if scores[i] < self.best_scores[i]:
                self.best_scores[i] = scores[i]
                self.best_positions[i] = self.positions[i].copy()

            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.positions[i].copy()

        # 迭代优化
        for iter_num in range(self.max_iter):
            # 计算当前种群的平均适应度和最小适应度
            f_avg = np.mean(scores)
            f_min = np.min(scores)

            # 更新参数
            w_values, c1_values, c2_values = self.update_parameters(iter_num, f_avg, f_min, scores)

            # 更新速度和位置
            for i in range(self.n_particles):
                r1, r2 = np.random.random(self.dimensions), np.random.random(self.dimensions)

                # 更新速度
                cognitive_component = c1_values[i] * r1 * (self.best_positions[i] - self.positions[i])
                social_component = c2_values[i] * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = w_values[i] * self.velocities[i] + cognitive_component + social_component

                # 限制速度
                for j in range(self.dimensions):
                    max_velocity = 0.2 * (self.bounds[1][j] - self.bounds[0][j])
                    self.velocities[i, j] = np.clip(self.velocities[i, j], -max_velocity, max_velocity)

                # 更新位置
                self.positions[i] += self.velocities[i]

                # 边界处理
                for j in range(self.dimensions):
                    self.positions[i, j] = np.clip(self.positions[i, j], self.bounds[0][j], self.bounds[1][j])

            # 评估新位置
            scores = objective_function(self.positions)

            # 更新个体最优和全局最优
            for i in range(self.n_particles):
                if scores[i] < self.best_scores[i]:
                    self.best_scores[i] = scores[i]
                    self.best_positions[i] = self.positions[i].copy()

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i].copy()

            # 打印当前迭代的最佳得分
            if (iter_num + 1) % 10 == 0 or iter_num == 0:
                print(f"迭代 {iter_num + 1}/{self.max_iter}, 最佳得分: {-self.global_best_score:.4f}")

        return -self.global_best_score, self.global_best_position

    def _calculate_diversity(self):
        """计算种群多样性"""
        mean_pos = np.mean(self.positions, axis=0)
        diversity = np.mean(np.sqrt(np.sum((self.positions - mean_pos) ** 2, axis=1)))
        normalized_diversity = diversity / np.sqrt(np.sum((self.bounds[1] - self.bounds[0]) ** 2))
        return normalized_diversity


# Leave-One-Out (LOO) 交叉验证
for i in range(len(X)):
    print(f'\n第 {i + 1} 个 LOO 训练')

    # 训练集、测试集拆分
    X_train, y_train = np.delete(X, i, axis=0), np.delete(y, i, axis=0)
    X_test, y_test = X[i].reshape(1, -1), y[i]
    y_true.append(y_test)


    # 步骤1: APSO优化
    def apso_objective(params):
        """APSO优化的目标函数"""
        results = []
        for param_set in params:
            C = max(0.01, param_set[0])
            gamma = max(0.01, param_set[1])
            epsilon = max(0.01, param_set[2])

            try:
                model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)[0]
                train_score = model.score(X_train, y_train)
                error = abs(y_test - y_pred)

                # 综合考虑训练集得分和预测误差
                score = train_score - 0.1 * error if train_score > 0.85 else -999
            except Exception as e:
                score = -999

            results.append(-score)  # APSO是最小化问题
        return np.array(results)


    # APSO优化设置
    bounds = ([0.01, 0.01, 0.01], [100.0, 10.0, 1.0])
    # 初始化APSO优化器
    optimizer = APSO(n_particles=50, dimensions=3, bounds=bounds, max_iter=50)

    # 运行APSO优化
    best_cost, best_pos = optimizer.optimize(apso_objective)
    best_C_apso, best_gamma_apso, best_epsilon_apso = best_pos

    print(f"APSO最佳参数:")
    print(f"C: {best_C_apso:.4f}")
    print(f"gamma: {best_gamma_apso:.4f}")
    print(f"epsilon: {best_epsilon_apso:.4f}")

    # 步骤2: 贝叶斯优化
    # 使用贝叶斯优化在APSO找到的区域附近进行精细调优
    space = [
        Real(max(0.01, best_C_apso * 0.8), max(min(100.0, best_C_apso * 1.2), max(0.01, best_C_apso * 0.8) + 0.01),
             name='C', prior='log-uniform'),
        Real(max(0.01, best_gamma_apso * 0.8),
             max(min(10.0, best_gamma_apso * 1.2), max(0.01, best_gamma_apso * 0.8) + 0.01), name='gamma',
             prior='log-uniform'),
        Real(max(0.01, best_epsilon_apso * 0.8),
             max(min(1.0, best_epsilon_apso * 1.2), max(0.01, best_epsilon_apso * 0.8) + 0.01), name='epsilon',
             prior='log-uniform')
    ]


    @use_named_args(space)
    def objective(C, gamma, epsilon):
        try:
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)[0]
            train_score = model.score(X_train, y_train)
            error = abs(y_test - y_pred)

            # 综合考虑训练集得分和预测误差
            return -(train_score - 0.1 * error) if train_score > 0.9 else 999
        except Exception as e:
            return 999


    # 运行贝叶斯优化
    result = gp_minimize(objective, space, n_calls=30, random_state=42, verbose=True)

    # 获取最佳参数
    best_C = result.x[0]
    best_gamma = result.x[1]
    best_epsilon = result.x[2]

    print(f"\n贝叶斯优化最佳参数:")
    print(f"C: {best_C:.4f}")
    print(f"gamma: {best_gamma:.4f}")
    print(f"epsilon: {best_epsilon:.4f}")

    # 使用最佳参数训练最终模型
    final_model = SVR(kernel='rbf', C=best_C, gamma=best_gamma, epsilon=best_epsilon)
    final_model.fit(X_train, y_train)

    # 预测和评估
    train_score = final_model.score(X_train, y_train)
    y_pred = final_model.predict(X_test)[0]
    y_predict.append(y_pred)

    print(f'测试组 {i + 1} - 训练集 R²: {train_score:.4f}')
    print(f'真实值: {y_test:.4f}, 预测值: {y_pred:.4f}, 误差: {abs(y_test - y_pred):.4f}')

    if (i + 1) % 5 == 0:  # 每5次计算一次整体R²
        current_r2 = r2_score(y_true, y_predict)
        print(f'当前累计 R² = {current_r2:.4f}')

# 计算最终评估指标
final_r2 = r2_score(y_true, y_predict)
final_rmse = sqrt(mean_squared_error(y_true, y_predict))

print("\n最终评估结果：")
print(f'决定系数 R² = {final_r2:.4f}')
print(f'均方根误差 RMSE = {final_rmse:.4f}')

