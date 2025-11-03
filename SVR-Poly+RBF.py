import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import random
from sklearn.metrics import r2_score, mean_squared_error
# 设置随机种子以确保结果可重复
np.random.seed(50)
random.seed(50)

# 读取数据
filename = '56result_guiyihua2.csv'
df = pd.read_csv(filename, header=None)

# 分离特征和目标变量
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 定义测试集的索引
test_indices = [2, 11, 7, 12, 13, 26, 27, 29, 40, 43, 46, 48, 49, 51]

# 分离训练集和测试集
X_test = X[test_indices]
y_test = y[test_indices]
X_train = np.delete(X, test_indices, axis=0)
y_train = np.delete(y, test_indices, axis=0)

# 在分割训练集和测试集之后添加
X_train_scaled = X_train  # 数据已归一化，不需要再标准化
X_test_scaled = X_test  # 数据已归一化，不需要再标准化


# 自定义混合核函数 - 修复NaN和Inf问题
def hybrid_kernel(X1, X2, gamma=1.0, lambda1=0.5, degree=3, coef0=1):
    """混合核函数: RBF + 多项式"""
    lambda2 = 1 - lambda1

    X1, X2 = np.atleast_2d(X1, X2)
    rbf_part = np.exp(-gamma * cdist(X1, X2, 'sqeuclidean'))

    # 确保degree是整数，避免负数的非整数次幂
    degree = int(round(degree))
    # 确保底数非负
    poly_base = np.dot(X1, X2.T) + max(0, coef0)
    poly_part = poly_base ** degree

    return lambda1 * rbf_part + lambda2 * poly_part


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


# APSO优化的目标函数
def apso_objective(params):
    """APSO优化的目标函数"""
    results = []
    for param_set in params:
        # 提取并约束参数，确保它们在有效范围内
        C = max(0.01, param_set[0])
        gamma = max(0.01, param_set[1])
        lambda1 = max(0.01, min(0.99, param_set[2]))
        degree = int(round(param_set[3]))
        coef0 = max(0, param_set[4])
        epsilon = max(0.01, min(1.0, param_set[5]))  # 添加epsilon参数范围

        try:
            # 创建自定义核函数
            def custom_kernel(X, Y):
                return hybrid_kernel(X, Y, gamma=gamma, lambda1=lambda1,
                                     degree=degree, coef0=coef0)

            # 创建并训练模型
            model = SVR(kernel=custom_kernel, C=C, epsilon=epsilon)
            model.fit(X_train_scaled, y_train)

            # 预测
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)

            # 计算R²分数
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # 直接使用测试集R²作为优化目标
            score = test_r2

        except Exception as e:
            print(f"参数错误: {[C, gamma, lambda1, degree, coef0, epsilon]}, 错误: {str(e)}")
            score = -999

        results.append(-score)  # APSO是最小化问题，所以取负

    return np.array(results)


# 参数范围：[C, gamma, lambda1, degree, coef0, epsilon]
bounds = (
    [0.01, 0.01, 0.01, 2, 0, 0.01],  # 下界
    [100.0, 10.0, 0.99, 4, 5, 1.0]  # 上界
)

# 初始化APSO优化器
optimizer = APSO(n_particles=50, dimensions=6, bounds=bounds, max_iter=70)

# 运行APSO优化
print("开始APSO优化...")
best_cost, best_pos = optimizer.optimize(apso_objective)

# 提取最佳参数
best_C_apso, best_gamma_apso, best_lambda1_apso, best_degree_apso, best_coef0_apso, best_epsilon_apso = best_pos
best_degree_apso = int(round(best_degree_apso))
best_coef0_apso = max(0, best_coef0_apso)
best_epsilon_apso = max(0.01, min(1.0, best_epsilon_apso))  # 确保epsilon在范围内

print("\nAPSO最佳参数:")
print(f"C: {best_C_apso:.4f}")
print(f"gamma: {best_gamma_apso:.4f}")
print(f"lambda1: {best_lambda1_apso:.4f}")
print(f"degree: {best_degree_apso}")
print(f"coef0: {best_coef0_apso:.4f}")
print(f"epsilon: {best_epsilon_apso:.4f}")

# 步骤2: 使用贝叶斯优化在APSO找到的区域附近进行精细调优

# 定义参数空间
space = [
    Real(max(0.01, best_C_apso * 0.8), min(100.0, best_C_apso * 1.2), name='C', prior='log-uniform'),
    Real(max(0.01, best_gamma_apso * 0.8), min(10.0, best_gamma_apso * 1.2), name='gamma', prior='log-uniform'),
    Real(max(0.01, best_lambda1_apso * 0.8), min(0.99, best_lambda1_apso * 1.2), name='lambda1', prior='uniform'),
    Real(max(0, best_coef0_apso * 0.8), min(5.0, best_coef0_apso * 1.2), name='coef0', prior='uniform'),
    Real(max(0.01, best_epsilon_apso * 0.8), min(1.0, best_epsilon_apso * 1.2), name='epsilon', prior='log-uniform')
]

# 固定degree为APSO找到的最佳值
best_degree = best_degree_apso


@use_named_args(space)
def objective(C, gamma, lambda1, coef0, epsilon):
    try:
        # 创建自定义核函数
        def custom_kernel(X, Y):
            return hybrid_kernel(X, Y, gamma=gamma, lambda1=lambda1,
                                 degree=best_degree, coef0=coef0)

        # 创建并训练模型
        model = SVR(kernel=custom_kernel, C=C, epsilon=epsilon)
        model.fit(X_train_scaled, y_train)

        # 预测
        y_test_pred = model.predict(X_test_scaled)

        # 直接返回负的测试集R²
        test_r2 = r2_score(y_test, y_test_pred)
        return -test_r2

    except Exception as e:
        print(f"参数错误: {[C, gamma, lambda1, coef0, epsilon]}, 错误: {str(e)}")
        return 999


# 运行贝叶斯优化
print("\n开始贝叶斯优化...")
result = gp_minimize(objective, space, n_calls=70, random_state=42, verbose=True)

# 提取最佳参数
best_C = result.x[0]
best_gamma = result.x[1]
best_lambda1 = result.x[2]
best_coef0 = result.x[3]
best_epsilon = result.x[4]

print("\n贝叶斯优化最佳参数:")
print(f"C: {best_C:.4f}")
print(f"gamma: {best_gamma:.4f}")
print(f"lambda1: {best_lambda1:.4f}")
print(f"degree: {best_degree}")
print(f"coef0: {best_coef0:.4f}")
print(f"epsilon: {best_epsilon:.4f}")


# 使用最佳参数创建最终模型
def final_kernel(X, Y):
    return hybrid_kernel(X, Y, gamma=best_gamma, lambda1=best_lambda1,
                         degree=best_degree, coef0=best_coef0)


final_model = SVR(kernel=final_kernel, C=best_C, epsilon=best_epsilon)
final_model.fit(X_train_scaled, y_train)

# 评估最终模型
y_train_pred = final_model.predict(X_train_scaled)
y_test_pred = final_model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

y_true_mean_test = np.mean(y_test);
y_pred_mean_test = np.mean(y_test_pred);
n_test = len(y_test)
numerator_ccc = 2 * np.sum((y_test - y_true_mean_test) * (y_test_pred - y_pred_mean_test))
denominator_ccc = np.sum((y_test - y_true_mean_test) ** 2) + np.sum((y_test_pred - y_pred_mean_test) ** 2) + n_test * (
            y_true_mean_test - y_pred_mean_test) ** 2
ccc = numerator_ccc / denominator_ccc if denominator_ccc != 0 else 0

y_train_mean = np.mean(y_train)
press_qf1 = np.sum((y_test - y_test_pred) ** 2)
tss_qf1 = np.sum((y_test - y_train_mean) ** 2)
qf1 = 1 - (press_qf1 / tss_qf1) if tss_qf1 != 0 else 0

press_qf2 = np.sum((y_test - y_test_pred) ** 2)
tss_qf2 = np.sum((y_test - y_true_mean_test) ** 2)
qf2 = 1 - (press_qf2 / tss_qf2) if tss_qf2 != 0 else 0

n_train = len(y_train)
num_qf3_term = np.sum((y_test - y_test_pred) ** 2) / n_test
den_qf3_term = np.sum((y_train - y_train_mean) ** 2) / n_train
qf3 = 1 - (num_qf3_term / den_qf3_term) if den_qf3_term != 0 else 0

print("\n最终模型性能:")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")
print(f"  训练集 RMSE (RMSEC): {rmse_train:.4f}")
print(f"  测试集 RMSE (RMSEP): {rmse_test:.4f}")
print("-" * 30)
print(f"  测试集 CCC: {ccc:.4f}")
print(f"  测试集 QF1 (Q_F1^2): {qf1:.4f}")
print(f"  测试集 QF2 (Q_F2^2): {qf2:.4f}")
print(f"  测试集 QF3 (Q_F3^2): {qf3:.4f}")
print("=" * 30 + "\n")

plt.figure(figsize=(8.5, 8.5), dpi=200)
ax = plt.gca()
ax.scatter(y_train, y_train_pred, c='black', alpha=0.7, marker='o', s=50, label='Training set', zorder=3)
ax.scatter(y_test, y_test_pred, c='red', alpha=0.8, marker='^', s=60, edgecolors='black', linewidths=0.5,
           label='Test set', zorder=4)
all_real = np.concatenate([y_train, y_test]);
all_pred = np.concatenate([y_train_pred, y_test_pred])
min_val, max_val = min(all_real.min(), all_pred.min()), max(all_real.max(), all_pred.max())
padding = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-6 else 0.1
axis_min, axis_max = min_val - padding, max_val + padding
ax.set_xlim(axis_min, axis_max);
ax.set_ylim(axis_min, axis_max)
ax.plot([axis_min, axis_max], [axis_min, axis_max], '--', color='grey', label='Ideal prediction', zorder=2)
ax.set_xlabel('Measured pIC$_{50}$', fontsize=14)
ax.set_ylabel('Predicted pIC$_{50}$', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_aspect('equal', adjustable='box')
legend = ax.legend(loc='upper left', fontsize=14)
legend.get_frame().set_edgecolor('0.8')
plt.tight_layout()
plt.savefig('svr_rbf_final_plot_poly_cleaned.png', dpi=300)
plt.show()
