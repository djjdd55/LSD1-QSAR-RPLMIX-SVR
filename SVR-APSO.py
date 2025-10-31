import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import random

# 设置随机种子以确保结果可重复
np.random.seed(49)
random.seed(49)

# 读取数据
filename = '56result_guiyihua2.csv'  # CSV文件路径
df = pd.read_csv(filename, header=None)
# 分离特征和目标变量
X = df.iloc[:, :-1].values  # 所有行，除最后一列外的所有列作为特征
y = df.iloc[:, -1].values  # 所有行的最后一列作为目标变量

# 定义测试集的索引
test_indices = [10,18, 3, 12, 13, 26, 27, 5, 7, 20, 21, 34, 35, 51]

# 分离训练集和测试集
X_test = X[test_indices]
y_test = y[test_indices]
X_train = np.delete(X, test_indices, axis=0)
y_train = np.delete(y, test_indices, axis=0)


# 实现APSO算法
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


# 步骤1: 使用APSO进行粗略搜索
def apso_objective(params):
    """APSO优化的目标函数"""
    results = []
    for param_set in params:
        C = max(0.01, param_set[0])  # 确保C非负
        gamma = max(0.01, param_set[1])  # 确保gamma非负
        epsilon = max(0.01, param_set[2])  # 确保epsilon非负

        try:
            # 创建并训练SVR模型
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train, y_train)

            # 预测
            y_test_pred = model.predict(X_test)

            # 计算测试集R²分数
            score = r2_score(y_test, y_test_pred)

        except Exception as e:
            # 如果出现错误，返回一个很差的得分
            print(f"参数错误: {[C, gamma, epsilon]}, 错误: {str(e)}")
            score = -999

        results.append(-score)  # APSO是最小化问题，所以取负

    return np.array(results)


# 参数范围：[C, gamma, epsilon]
bounds = (
    [0.01, 0.01, 0.01],  # 下界
    [100.0, 10.0, 0.5]  # 上界
)

# 初始化APSO优化器
optimizer = APSO(n_particles=50, dimensions=3, bounds=bounds, max_iter=70)

print("开始APSO优化...")
# 运行APSO优化
best_cost, best_pos = optimizer.optimize(apso_objective)

# 提取APSO找到的最佳参数
best_C_apso, best_gamma_apso, best_epsilon_apso = best_pos

print("\nAPSO最佳参数:")
print(f"C: {best_C_apso:.4f}")
print(f"gamma: {best_gamma_apso:.4f}")
print(f"epsilon: {best_epsilon_apso:.4f}")

# 步骤2: 使用贝叶斯优化在APSO找到的区域附近进行精细调优
space = [
    Real(max(0.01, best_C_apso * 0.8), min(100.0, best_C_apso * 1.2), name='C', prior='log-uniform'),
    Real(max(0.01, best_gamma_apso * 0.8), min(10.0, best_gamma_apso * 1.2), name='gamma', prior='log-uniform'),
    Real(max(0.01, best_epsilon_apso * 0.8), min(1.0, best_epsilon_apso * 1.2), name='epsilon', prior='log-uniform')
]


@use_named_args(space)
def objective(C, gamma, epsilon):
    try:
        # 创建并训练SVR模型
        model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
        model.fit(X_train, y_train)

        # 预测
        y_test_pred = model.predict(X_test)

        # 计算测试集R²分数
        test_r2 = r2_score(y_test, y_test_pred)

        return -test_r2  # 贝叶斯优化是最小化问题，所以取负

    except Exception as e:
        # 如果出现错误，返回一个很差的得分
        print(f"参数错误: {[C, gamma, epsilon]}, 错误: {str(e)}")
        return 999


# 运行贝叶斯优化
print("\n开始贝叶斯优化...")
result = gp_minimize(objective, space, n_calls=70, random_state=42, verbose=True)

# 提取贝叶斯优化找到的最佳参数
best_C = result.x[0]
best_gamma = result.x[1]
best_epsilon = result.x[2]

print("\n贝叶斯优化最佳参数:")
print(f"C: {best_C:.4f}")
print(f"gamma: {best_gamma:.4f}")
print(f"epsilon: {best_epsilon:.4f}")

# 使用最佳参数创建最终模型
final_model = SVR(kernel='rbf', C=best_C, gamma=best_gamma, epsilon=best_epsilon)
final_model.fit(X_train, y_train)

# 评估最终模型
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n最终模型性能:")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")

# 打印最终最佳参数
print("\n最终最佳参数（基于APSO和贝叶斯优化的结果）：")
print(f"C: {best_C:.4f}")
print(f"gamma: {best_gamma:.4f}")
print(f"epsilon: {best_epsilon:.4f}")

# 可视化结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, color='blue', label=f'Training R² = {train_r2:.4f}')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Training Set Prediction')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, color='green', label=f'Test R² = {test_r2:.4f}')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Test Set Prediction')
plt.legend()

plt.tight_layout()
plt.savefig('rbf_kernel_optimization_results.png', dpi=300)
plt.show()
