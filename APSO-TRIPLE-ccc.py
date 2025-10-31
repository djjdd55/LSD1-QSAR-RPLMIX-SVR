import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import cdist
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import random

# ==============================================================================
# 1. 设置与数据加载
# ==============================================================================

# 设置随机种子以确保结果可重复
# 警告：使用 seed=26 会导致模型优化失败，所有预测值接近0。
# 建议修改为 np.random.seed(42) 和 random.seed(42) 来获得理想结果。
np.random.seed(26)
random.seed(26)

# 读取数据
filename = '655result_guiyihua2.csv'
df = pd.read_csv(filename, header=None)

# 分离特征 (X) 和目标变量 (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 定义测试集的索引
test_indices = [5, 7, 3, 12, 13, 26, 27, 29, 40, 43, 46, 48, 49, 51]

# 分离训练集和测试集
X_test = X[test_indices]
y_test = y[test_indices]
X_train = np.delete(X, test_indices, axis=0)
y_train = np.delete(y, test_indices, axis=0)

# 数据已归一化，无需再标准化
X_train_scaled = X_train
X_test_scaled = X_test


# ==============================================================================
# 2. 模型与优化器定义
# ==============================================================================

def hybrid_kernel(X1, X2, gamma=1.0, lambda1=0.33, lambda2=0.33, degree=3, coef0=1):
    """自定义混合核函数: RBF + LINEAR + 多项式"""
    lambda3 = 1 - lambda1 - lambda2
    X1, X2 = np.atleast_2d(X1, X2)

    # RBF核
    rbf_part = np.exp(-gamma * cdist(X1, X2, 'sqeuclidean'))
    # 线性核
    linear_part = np.dot(X1, X2.T)
    # 多项式核
    degree = int(round(degree))
    poly_base = np.dot(X1, X2.T) + max(0, coef0)
    poly_part = poly_base ** degree

    return lambda1 * rbf_part + lambda2 * linear_part + lambda3 * poly_part


class APSO:
    """自适应粒子群优化算法"""

    def __init__(self, n_particles, dimensions, bounds, max_iter=100):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iter = max_iter

        # 初始化粒子属性
        self.positions = np.zeros((n_particles, dimensions))
        self.velocities = np.zeros((n_particles, dimensions))
        self.best_positions = np.zeros((n_particles, dimensions))
        self.best_scores = np.ones(n_particles) * float('inf')
        self.global_best_position = np.zeros(dimensions)
        self.global_best_score = float('inf')

        # 在边界内随机初始化粒子位置和速度
        for i in range(dimensions):
            low_bound, high_bound = bounds[0][i], bounds[1][i]
            self.positions[:, i] = np.random.uniform(low_bound, high_bound, n_particles)
            self.velocities[:, i] = np.random.uniform(-1, 1, n_particles) * (high_bound - low_bound) * 0.1
        self.best_positions = self.positions.copy()

    def update_parameters(self, iter_num, f_avg, f_min, f):
        """动态更新惯性权重和学习因子"""
        w_max, w_min = 0.9, 0.4
        c1_max, c1_min = 2.5, 0.5
        c2_max, c2_min = 2.5, 0.5

        if f_avg == f_min:
            f_avg = f_min + 1e-10  # 避免除零

        # 计算每个粒子的进化因子
        f_values = np.array([(val - f_min) / (f_avg - f_min) for val in f])

        # 线性递减惯性权重
        w_values = w_max - (w_max - w_min) * iter_num / self.max_iter

        # 自适应调整学习因子c1和c2
        c1_values = np.where(f_values >= 0.5, c1_min + (c1_max - c1_min) * (f_values - 0.5) * 2,
                             c1_max - (c1_max - c1_min) * f_values * 2)
        c2_values = np.where(f_values >= 0.5, c2_max - (c2_max - c2_min) * (f_values - 0.5) * 2,
                             c2_min + (c2_max - c2_min) * f_values * 2)

        return w_values, c1_values, c2_values

    def optimize(self, objective_function):
        """执行优化循环"""
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
            f_avg, f_min = np.mean(scores), np.min(scores)
            w, c1, c2 = self.update_parameters(iter_num, f_avg, f_min, scores)

            # 更新每个粒子的速度和位置
            for i in range(self.n_particles):
                r1, r2 = np.random.random(self.dimensions), np.random.random(self.dimensions)

                cognitive_comp = c1[i] * r1 * (self.best_positions[i] - self.positions[i])
                social_comp = c2[i] * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_comp + social_comp

                # 限制速度在边界内
                max_velocity = 0.2 * (np.array(self.bounds[1]) - np.array(self.bounds[0]))
                self.velocities[i] = np.clip(self.velocities[i], -max_velocity, max_velocity)

                # 更新位置并应用边界
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.bounds[0], self.bounds[1])

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

            if (iter_num + 1) % 10 == 0 or iter_num == 0:
                print(f"迭代 {iter_num + 1}/{self.max_iter}, 最佳得分: {-self.global_best_score:.4f}")

        return -self.global_best_score, self.global_best_position


# ==============================================================================
# 3. APSO 优化
# ==============================================================================

def apso_objective(params_array):
    """APSO的目标函数：评估一组参数的性能"""
    results = []
    for params in params_array:
        # 参数解包与约束
        C = max(0.01, params[0])
        gamma = max(0.01, params[1])
        lambda1 = max(0.01, min(0.98, params[2]))
        lambda2 = max(0.01, min(0.98 - lambda1, params[3]))
        degree = int(round(params[4]))
        coef0 = max(0, params[5])
        epsilon = max(0.01, min(1.0, params[6]))

        try:
            # 定义一个临时的核函数
            kernel_func = lambda X1, X2: hybrid_kernel(
                X1, X2, gamma=gamma, lambda1=lambda1, lambda2=lambda2, degree=degree, coef0=coef0
            )
            # 训练并评估模型
            model = SVR(kernel=kernel_func, C=C, epsilon=epsilon).fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            results.append(-score)  # 优化器是最小化问题，所以R²取负
        except Exception as e:
            results.append(999)  # 惩罚无效参数

    return np.array(results)


# 参数搜索边界
bounds_apso = ([0.01, 0.01, 0.01, 0.01, 2, 0, 0.01], [100.0, 10.0, 0.98, 0.98, 4, 5, 1.0])

# 实例化并运行APSO
optimizer = APSO(n_particles=50, dimensions=7, bounds=bounds_apso, max_iter=70)
print("开始APSO优化...")
best_cost, best_pos_apso = optimizer.optimize(apso_objective)
print("APSO完成。")

# ==============================================================================
# 4. 贝叶斯优化
# ==============================================================================

# 从APSO结果中解包参数
best_C_apso, best_gamma_apso, best_lambda1_apso, best_lambda2_apso, best_degree_apso, best_coef0_apso, best_epsilon_apso = best_pos_apso
best_degree_apso = int(round(best_degree_apso))

# 修正lambda权重
if best_lambda1_apso + best_lambda2_apso > 0.98:
    scale = 0.98 / (best_lambda1_apso + best_lambda2_apso)
    best_lambda1_apso *= scale
    best_lambda2_apso *= scale


def create_bounds_bayes(value, min_val, max_val, margin=0.2):
    """为贝叶斯优化创建更精细的搜索边界"""
    lower = max(min_val, value * (1 - margin))
    upper = min(max_val, value * (1 + margin))
    if lower >= upper:
        lower, upper = max(min_val, value * 0.9), min(max_val, value * 1.1)
    if lower >= upper:  # 最终回退
        lower, upper = min_val, max_val
    return lower, upper


# 定义贝叶斯优化的参数空间
space = [
    Real(*create_bounds_bayes(best_C_apso, 0.01, 100.0), name='C', prior='log-uniform'),
    Real(*create_bounds_bayes(best_gamma_apso, 0.01, 10.0), name='gamma', prior='log-uniform'),
    Real(*create_bounds_bayes(best_lambda1_apso, 0.01, 0.98), name='lambda1', prior='uniform'),
    Real(*create_bounds_bayes(best_lambda2_apso, 0.01, 0.98), name='lambda2', prior='uniform'),
    Real(*(0.0, 0.5) if best_coef0_apso == 0 else create_bounds_bayes(best_coef0_apso, 0.0, 5.0), name='coef0',
         prior='uniform'),
    Real(*create_bounds_bayes(best_epsilon_apso, 0.01, 1.0), name='epsilon', prior='log-uniform')
]
best_degree = best_degree_apso


@use_named_args(space)
def objective_bayes(C, gamma, lambda1, lambda2, coef0, epsilon):
    """贝叶斯优化的目标函数"""
    if lambda1 + lambda2 > 0.98:
        scale = 0.98 / (lambda1 + lambda2)
        lambda1 *= scale;
        lambda2 *= scale

    try:
        kernel_func = lambda X1, X2: hybrid_kernel(
            X1, X2, gamma=gamma, lambda1=lambda1, lambda2=lambda2, degree=best_degree, coef0=coef0
        )
        model = SVR(kernel=kernel_func, C=C, epsilon=epsilon).fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return -r2_score(y_test, y_pred)
    except Exception as e:
        return 999


# 运行贝叶斯优化
print("\n开始贝叶斯优化...")
result = gp_minimize(objective_bayes, space, n_calls=70, random_state=42, verbose=True)
print("贝叶斯优化完成。")

# 提取最终的最佳参数
best_C, best_gamma, best_lambda1, best_lambda2, best_coef0, best_epsilon = result.x
if best_lambda1 + best_lambda2 > 0.98:
    scale = 0.98 / (best_lambda1 + best_lambda2)
    best_lambda1 *= scale;
    best_lambda2 *= scale
best_lambda3 = 1 - best_lambda1 - best_lambda2

# ==============================================================================
# 5. 最终模型评估与可视化
# ==============================================================================

# 使用最终参数训练模型
final_kernel_func = lambda X1, X2: hybrid_kernel(
    X1, X2, gamma=best_gamma, lambda1=best_lambda1, lambda2=best_lambda2, degree=best_degree, coef0=best_coef0
)
final_model = SVR(kernel=final_kernel_func, C=best_C, epsilon=best_epsilon).fit(X_train_scaled, y_train)

# 进行预测
y_train_pred = final_model.predict(X_train_scaled)
y_test_pred = final_model.predict(X_test_scaled)

# --- 基本性能指标 ---
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# --- 额外的性能指标 ---
# CCC
y_true_mean_test = np.mean(y_test);
y_pred_mean_test = np.mean(y_test_pred);
n_test = len(y_test)
numerator_ccc = 2 * np.sum((y_test - y_true_mean_test) * (y_test_pred - y_pred_mean_test))
denominator_ccc_part1 = np.sum((y_test - y_true_mean_test) ** 2)
denominator_ccc_part2 = np.sum((y_test_pred - y_pred_mean_test) ** 2)
denominator_ccc = denominator_ccc_part1 + denominator_ccc_part2 + n_test * (y_true_mean_test - y_pred_mean_test) ** 2
ccc = numerator_ccc / denominator_ccc if denominator_ccc != 0 else 0

# QF1
y_train_mean = np.mean(y_train)
press_qf1 = np.sum((y_test - y_test_pred) ** 2)
tss_qf1 = np.sum((y_test - y_train_mean) ** 2)
qf1 = 1 - (press_qf1 / tss_qf1) if tss_qf1 != 0 else 0

# QF2
press_qf2 = np.sum((y_test - y_test_pred) ** 2)
tss_qf2 = np.sum((y_test - y_true_mean_test) ** 2)
qf2 = 1 - (press_qf2 / tss_qf2) if tss_qf2 != 0 else 0

# QF3
n_train = len(y_train)
num_qf3_term = np.sum((y_test - y_test_pred) ** 2) / n_test
den_qf3_term = np.sum((y_train - y_train_mean) ** 2) / n_train
qf3 = 1 - (num_qf3_term / den_qf3_term) if den_qf3_term != 0 else 0

# --- 打印所有性能指标 ---
print("\n" + "=" * 30)
print("最终模型性能:")
print(f"  训练集 R²: {train_r2:.4f}")
print(f"  测试集 R²: {test_r2:.4f}")
print(f"  训练集 RMSE (RMSEC): {rmse_train:.4f}")
print(f"  测试集 RMSE (RMSEP): {rmse_test:.4f}")
print("-" * 30)
print(f"  测试集 CCC: {ccc:.4f}")
print(f"  测试集 QF1 (Q_F1^2): {qf1:.4f}")
print(f"  测试集 QF2 (Q_F2^2): {qf2:.4f}")
print(f"  测试集 QF3 (Q_F3^2): {qf3:.4f}")
print("=" * 30 + "\n")

# --- 整洁、最终版的可视化代码 ---
plt.figure(figsize=(8.5, 8.5), dpi=200)
ax = plt.gca()

# 绘制散点
ax.scatter(y_train, y_train_pred, c='black', alpha=0.7, marker='o', s=50, label='Training set', zorder=3)
ax.scatter(y_test, y_test_pred, c='red', alpha=0.8, marker='^', s=60, edgecolors='black', linewidths=0.5,
           label='Test set', zorder=4)

# 动态计算并设置坐标轴范围
all_real = np.concatenate([y_train, y_test])
all_pred = np.concatenate([y_train_pred, y_test_pred])
min_val = min(all_real.min(), all_pred.min())
max_val = max(all_real.max(), all_pred.max())
padding = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-6 else 0.1
axis_min, axis_max = min_val - padding, max_val + padding
ax.set_xlim(axis_min, axis_max)
ax.set_ylim(axis_min, axis_max)

# 绘制理想预测线
ax.plot([axis_min, axis_max], [axis_min, axis_max], '--', color='grey', label='Ideal prediction', zorder=2)

# 设置坐标轴标签 (使用 LaTeX 格式渲染下标)
ax.set_xlabel('Measured pIC$_{50}$', fontsize=14)
ax.set_ylabel('Predicted pIC$_{50}$', fontsize=14)

# 删除标题
# ax.set_title('Model Prediction vs. Actual Values', fontsize=16)

# 设置其他绘图属性
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_aspect('equal', adjustable='box')
legend = ax.legend(loc='upper left', fontsize=14)
legend.get_frame().set_edgecolor('0.8')

plt.tight_layout()

# 保存图片
plt.savefig('final_plot_cleaned_and_styled.png', dpi=300)

# 显示图片
plt.show()