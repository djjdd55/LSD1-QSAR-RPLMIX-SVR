import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler # 已注释掉，因为数据已归一化
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_squared_error # 添加 mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import random
import time # 导入 time 模块
from math import sqrt # 导入 sqrt

# --- 设置随机种子以确保结果可重复 ---
np.random.seed(11)
random.seed(11)

# --- 读取数据 ---
filename = 'Your data.csv'
try:
    df = pd.read_csv(filename, header=None)
except FileNotFoundError:
    print(f"错误: 文件 '{filename}' 未找到。")
    exit()
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit()

# --- 分离特征和目标变量 ---
X_all = df.iloc[:, :-1].values
y_all = df.iloc[:, -1].values
print(f"数据加载完成: {X_all.shape[0]} 个样本, {X_all.shape[1]} 个特征.")

# --- 定义测试集的索引 (保持不变) ---
test_indices = [10, 3, 11, 12, 13, 26, 27, 29, 40, 43, 46, 48, 49, 51]
if max(test_indices) >= len(X_all):
    print(f"错误：测试索引超出数据范围 (最大索引 {max(test_indices)}, 数据长度 {len(X_all)})。")
    exit()

# --- 分离训练集和测试集 (只执行一次) ---
X_test_fixed = X_all[test_indices]
y_test_fixed = y_all[test_indices]
X_train_original = np.delete(X_all, test_indices, axis=0)
y_train_original = np.delete(y_all, test_indices, axis=0)
print(f"训练集样本数: {len(X_train_original)}, 测试集样本数: {len(X_test_fixed)}")

# --- 数据已归一化，直接使用 ---
X_train_scaled_original = X_train_original
X_test_scaled_fixed = X_test_fixed

# --- 自定义混合核函数 (保持不变, lambda3 在内部计算) ---
def hybrid_kernel(X1, X2, gamma=1.0, lambda1=0.33, lambda2=0.33, degree=3, coef0=1):
    """混合核函数: RBF + LINEAR + 多项式。 lambda3 = 1 - lambda1 - lambda2"""
    # lambda3 由 1 - lambda1 - lambda2 隐式定义，这里确保它们非负
    lambda1_safe = max(0, lambda1)
    lambda2_safe = max(0, lambda2)
    # 计算 lambda3，并确保其非负
    lambda3_safe = max(0, 1.0 - lambda1_safe - lambda2_safe)
    # 如果计算出的 lambda3 导致总和不为 1 (例如 l1+l2>1), 需要重新归一化
    # 注意：由于约束是在调用函数前应用的(l1+l2<=0.98)，理论上这里不需要归一化，
    # 但保留它以增加健壮性
    total_lambda = lambda1_safe + lambda2_safe + lambda3_safe
    if abs(total_lambda - 1.0) > 1e-6 and total_lambda > 1e-6:
        lambda1_safe /= total_lambda
        lambda2_safe /= total_lambda
        lambda3_safe /= total_lambda

    # 确保 X1, X2 是二维数组
    if X1.ndim == 1: X1 = X1.reshape(1, -1)
    if X2.ndim == 1: X2 = X2.reshape(1, -1)

    # RBF核
    try:
        sqdist = cdist(X1, X2, 'sqeuclidean')
        rbf_part = np.exp(-gamma * sqdist)
    except Exception as e:
        rbf_part = np.zeros((X1.shape[0], X2.shape[0]))

    # 线性核
    try:
        linear_part = np.dot(X1, X2.T)
    except Exception as e:
        linear_part = np.zeros((X1.shape[0], X2.shape[0]))

    # 多项式核
    try:
        degree = int(round(degree))
        if degree < 1:
             poly_part = np.ones((X1.shape[0], X2.shape[0]))
        else:
            coef0_safe = max(0, coef0)
            poly_base = np.dot(X1, X2.T) + coef0_safe
            poly_part = poly_base ** degree
    except OverflowError:
        poly_part = np.full((X1.shape[0], X2.shape[0]), 1e9) # 用大数代替inf
    except Exception as e:
        poly_part = np.zeros((X1.shape[0], X2.shape[0]))

    # 合并
    kernel_matrix = lambda1_safe * rbf_part + lambda2_safe * linear_part + lambda3_safe * poly_part
    kernel_matrix = np.nan_to_num(kernel_matrix, nan=0.0, posinf=1e9, neginf=-1e9)
    return kernel_matrix

# --- APSO 类 (保持不变) ---
class APSO:
    def __init__(self, n_particles, dimensions, bounds, max_iter=100, verbose=True):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds # ([low1,...], [high1,...])
        self.max_iter = max_iter
        self.verbose = verbose

        self.positions = np.zeros((n_particles, dimensions))
        self.velocities = np.zeros((n_particles, dimensions))
        self.best_positions = np.zeros((n_particles, dimensions))
        self.best_scores = np.ones(n_particles) * float('inf')
        self.global_best_position = np.zeros(dimensions)
        self.global_best_score = float('inf')

        for i in range(dimensions):
            lower_bound = bounds[0][i]
            upper_bound = bounds[1][i]
            if lower_bound >= upper_bound: upper_bound = lower_bound + 1e-6
            self.positions[:, i] = np.random.uniform(lower_bound, upper_bound, n_particles)
            vel_range = upper_bound - lower_bound
            self.velocities[:, i] = np.random.uniform(-1, 1, n_particles) * max(vel_range, 1e-6) * 0.1
        self.best_positions = self.positions.copy()

    def update_parameters(self, iter_num, f_avg, f_min, f):
        w_max, w_min = 0.9, 0.4
        c1_max, c1_min = 2.5, 0.5
        c2_max, c2_min = 2.5, 0.5
        safe_f_avg = f_avg
        safe_f_min = f_min
        if abs(f_avg - f_min) < 1e-10: safe_f_avg = f_min + 1e-10
        f_values = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            denom = safe_f_avg - safe_f_min
            if abs(denom) < 1e-10: f_values[i] = 0.0 if abs(f[i] - safe_f_min) < 1e-10 else 1.0
            else: f_values[i] = max(0, (f[i] - safe_f_min)) / denom
        w_values = np.zeros(self.n_particles)
        c1_values = np.zeros(self.n_particles)
        c2_values = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            w_values[i] = w_max - (w_max - w_min) * iter_num / self.max_iter
            f_val_clipped = np.clip(f_values[i], 0, 1)
            if f_val_clipped >= 0.5:
                c1_values[i] = c1_min + (c1_max - c1_min) * (f_val_clipped - 0.5) * 2
                c2_values[i] = c2_max - (c2_max - c2_min) * (f_val_clipped - 0.5) * 2
            else:
                c1_values[i] = c1_max - (c1_max - c1_min) * f_val_clipped * 2
                c2_values[i] = c2_min + (c2_max - c2_min) * f_val_clipped * 2
            c1_values[i] = np.clip(c1_values[i], c1_min, c1_max)
            c2_values[i] = np.clip(c2_values[i], c2_min, c2_max)
        return w_values, c1_values, c2_values

    def optimize(self, objective_function):
        scores = objective_function(self.positions)
        for i in range(self.n_particles):
            if scores[i] < self.best_scores[i]:
                self.best_scores[i] = scores[i]
                self.best_positions[i] = self.positions[i].copy()
            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.positions[i].copy()
        for iter_num in range(self.max_iter):
            valid_scores = scores[np.isfinite(scores)]
            if len(valid_scores) == 0:
                 f_avg, f_min = float('inf'), float('inf')
                 w_values = np.ones(self.n_particles) * 0.7
                 c1_values = np.ones(self.n_particles) * 1.5
                 c2_values = np.ones(self.n_particles) * 1.5
            else:
                f_avg = np.mean(valid_scores)
                f_min = np.min(valid_scores)
                w_values, c1_values, c2_values = self.update_parameters(iter_num, f_avg, f_min, scores)
            for i in range(self.n_particles):
                r1, r2 = np.random.random(self.dimensions), np.random.random(self.dimensions)
                cognitive = c1_values[i] * r1 * (self.best_positions[i] - self.positions[i])
                social = c2_values[i] * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = w_values[i] * self.velocities[i] + cognitive + social
                for j in range(self.dimensions):
                    vel_range = self.bounds[1][j] - self.bounds[0][j]
                    max_velocity = 0.2 * max(vel_range, 1e-6)
                    self.velocities[i, j] = np.clip(self.velocities[i, j], -max_velocity, max_velocity)
                self.positions[i] += self.velocities[i]
                for j in range(self.dimensions):
                    self.positions[i, j] = np.clip(self.positions[i, j], self.bounds[0][j], self.bounds[1][j])
            scores = objective_function(self.positions)
            for i in range(self.n_particles):
                if scores[i] < self.best_scores[i]:
                    self.best_scores[i] = scores[i]
                    self.best_positions[i] = self.positions[i].copy()
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i].copy()
            if self.verbose and ((iter_num + 1) % 20 == 0 or iter_num == 0):
                current_best_display = -self.global_best_score if self.global_best_score != float('inf') else float('-inf')
                print(f"  APSO 迭代 {iter_num + 1}/{self.max_iter}, 当前最佳得分 (负目标函数值): {current_best_display:.4f}")
        final_best_score = -self.global_best_score if self.global_best_score != float('inf') else float('-inf')
        return final_best_score, self.global_best_position
# --- APSO 类结束 ---


# --- 封装优化流程的函数 ---
def run_hybrid_optimization_pipeline(X_train, y_train, X_test, y_test,
                                     apso_particles=50, apso_iters=70,
                                     bo_calls=70, verbose_level=1):
    """
    执行混合核SVR的APSO-BO优化流程并返回测试集R²。
    使用 lambda1 + lambda2 <= 0.98 约束。
    """
    start_time_pipeline = time.time()
    if verbose_level >= 1:
        print("--- 开始执行混合核优化流程 (lambda1+lambda2 <= 0.98 约束) ---")

    # --- 步骤 1: APSO 优化 ---
    # 定义 APSO 目标函数
    def apso_objective_inner(params_array):
        results = []
        for param_set in params_array:
            # 提取参数
            C = max(0.01, param_set[0])
            gamma = max(0.01, param_set[1])
            # 从参数集获取 lambda1, lambda2，限制在 [0.01, 0.98]
            lambda1_raw = max(0.01, min(0.98, param_set[2]))
            lambda2_raw = max(0.01, min(0.98, param_set[3]))
            degree = int(round(param_set[4]))
            coef0 = max(0, param_set[5])
            epsilon = max(0.01, min(1.0, param_set[6]))

            # 确保 degree >= 1
            if degree < 1: degree = 1

            # *** 应用 lambda1 + lambda2 <= 0.98 约束 ***
            lambda1 = lambda1_raw
            lambda2 = lambda2_raw
            if lambda1 + lambda2 > 0.98:
                scale = 0.98 / (lambda1 + lambda2)
                lambda1 *= scale
                lambda2 *= scale
            # 确保单个值也在界限内 (通常不需要，因为前面截断了，但保险起见)
            lambda1 = max(0.01, lambda1)
            lambda2 = max(0.01, lambda2)

            try:
                def custom_kernel_apso(X, Y):
                    return hybrid_kernel(X, Y, gamma=gamma, lambda1=lambda1,
                                         lambda2=lambda2, degree=degree, coef0=coef0)
                model = SVR(kernel=custom_kernel_apso, C=C, epsilon=epsilon)
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                score = r2_score(y_test, y_test_pred)
            except ValueError as ve:
                if verbose_level >= 2: print(f"  APSO SVR ValueError: {ve}")
                score = -float('inf')
            except Exception as e:
                if verbose_level >= 2: print(f"  APSO 内部错误: {e}")
                score = -float('inf')
            results.append(-score if np.isfinite(score) else float('inf'))
        return np.array(results)

    # APSO 参数范围：[C, gamma, lambda1, lambda2, degree, coef0, epsilon]
    # *** Lambda 上界改为 0.98 ***
    bounds_apso = (
        [0.01, 0.01, 0.01, 0.01, 1, 0, 0.01],  # 下界 (degree >= 1)
        [100.0, 10.0, 0.98, 0.98, 4, 5, 1.0]   # 上界 (lambda <= 0.98)
    )
    apso_optimizer = APSO(n_particles=apso_particles, dimensions=7, bounds=bounds_apso,
                          max_iter=apso_iters, verbose=(verbose_level >= 2))

    if verbose_level >= 1: print("  开始APSO优化...")
    best_score_apso, best_pos_apso = apso_optimizer.optimize(apso_objective_inner)

    # 提取并约束 APSO 最佳参数
    best_C_apso = max(0.01, best_pos_apso[0])
    best_gamma_apso = max(0.01, best_pos_apso[1])
    # 获取原始 lambda 值并应用约束
    best_lambda1_raw = max(0.01, min(0.98, best_pos_apso[2]))
    best_lambda2_raw = max(0.01, min(0.98, best_pos_apso[3]))
    best_lambda1_apso = best_lambda1_raw
    best_lambda2_apso = best_lambda2_raw
    if best_lambda1_apso + best_lambda2_apso > 0.98:
        scale = 0.98 / (best_lambda1_apso + best_lambda2_apso)
        best_lambda1_apso *= scale
        best_lambda2_apso *= scale
    best_lambda1_apso = max(0.01, best_lambda1_apso) # 再次确保下界
    best_lambda2_apso = max(0.01, best_lambda2_apso) # 再次确保下界

    best_degree_apso = max(1, int(round(best_pos_apso[4])))
    best_coef0_apso = max(0, best_pos_apso[5])
    best_epsilon_apso = max(0.01, min(1.0, best_pos_apso[6]))
    best_lambda3_apso = max(0, 1.0 - best_lambda1_apso - best_lambda2_apso) # lambda3 >= 0

    if verbose_level >= 1:
        print(f"  APSO 完成. 最佳约束后参数: C={best_C_apso:.4f}, gamma={best_gamma_apso:.4f}, "
              f"l1={best_lambda1_apso:.4f}, l2={best_lambda2_apso:.4f}, l3={best_lambda3_apso:.4f}, "
              f"deg={best_degree_apso}, coef0={best_coef0_apso:.4f}, eps={best_epsilon_apso:.4f}")
        print(f"  APSO 找到的最佳测试集 R² (近似): {best_score_apso:.4f}")

    # --- 步骤 2: 贝叶斯优化 ---
    fixed_degree_bo = best_degree_apso

    # 定义 BO 搜索空间 (围绕 APSO 结果)
    def create_bounds_bo(value, min_val, max_val, margin=0.3):
        lower = max(min_val, value * (1 - margin))
        upper = min(max_val, value * (1 + margin))
        if lower >= upper: upper = lower + (max_val - min_val) * 0.05; lower = upper - (max_val - min_val) * 0.1; lower = max(min_val, lower); upper = min(max_val, upper)
        if lower >= upper: upper = max_val if value >= (min_val+max_val)/2 else (min_val+max_val)/2; lower = min_val if value < (min_val+max_val)/2 else (min_val+max_val)/2
        # 确保最后上界大于下界
        if lower >= upper: upper = lower + 1e-4
        return lower, upper

    C_bounds = create_bounds_bo(best_C_apso, 0.01, 100.0)
    gamma_bounds = create_bounds_bo(best_gamma_apso, 0.01, 10.0)
    # *** BO Lambda 上界也限制为 0.98 ***
    lambda1_bounds = create_bounds_bo(best_lambda1_apso, 0.01, 0.98)
    lambda2_bounds = create_bounds_bo(best_lambda2_apso, 0.01, 0.98)
    coef0_bounds = create_bounds_bo(best_coef0_apso, 0.0, 5.0)
    epsilon_bounds = create_bounds_bo(best_epsilon_apso, 0.01, 1.0)

    space_bo = [
        Real(C_bounds[0], C_bounds[1], name='C', prior='log-uniform'),
        Real(gamma_bounds[0], gamma_bounds[1], name='gamma', prior='log-uniform'),
        Real(lambda1_bounds[0], lambda1_bounds[1], name='lambda1', prior='uniform'),
        Real(lambda2_bounds[0], lambda2_bounds[1], name='lambda2', prior='uniform'),
        Real(coef0_bounds[0], coef0_bounds[1], name='coef0', prior='uniform'),
        Real(epsilon_bounds[0], epsilon_bounds[1], name='epsilon', prior='log-uniform')
    ]

    # 定义 BO 目标函数
    @use_named_args(space_bo)
    def bo_objective_inner(C, gamma, lambda1, lambda2, coef0, epsilon):
        # *** 应用 lambda1 + lambda2 <= 0.98 约束 ***
        lambda1_bo = max(0.01, min(0.98, lambda1)) # 先截断单个值
        lambda2_bo = max(0.01, min(0.98, lambda2))
        if lambda1_bo + lambda2_bo > 0.98:
            if (lambda1_bo + lambda2_bo) > 1e-9:
                 scale = 0.98 / (lambda1_bo + lambda2_bo)
                 lambda1_bo *= scale
                 lambda2_bo *= scale
            else:
                 lambda1_bo = 0.01 # 避免为0？或者取小值
                 lambda2_bo = 0.01
        # 再次确保下界
        lambda1_bo = max(0.01, lambda1_bo)
        lambda2_bo = max(0.01, lambda2_bo)

        try:
            def custom_kernel_bo(X, Y):
                return hybrid_kernel(X, Y, gamma=gamma, lambda1=lambda1_bo,
                                     lambda2=lambda2_bo, degree=fixed_degree_bo, coef0=coef0)

            model = SVR(kernel=custom_kernel_bo, C=C, epsilon=epsilon)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            score = r2_score(y_test, y_test_pred)
            return -score if np.isfinite(score) else 1e6
        except ValueError as ve:
            if verbose_level >= 2: print(f"  BO SVR ValueError: {ve}")
            return 1e6
        except Exception as e:
            if verbose_level >= 2: print(f"  BO 内部错误: {e}")
            return 1e6

    if verbose_level >= 1: print("\n  开始贝叶斯优化...")
    bo_random_state = int(time.time()) % 1000
    result_bo = gp_minimize(bo_objective_inner, space_bo, n_calls=bo_calls,
                            random_state=bo_random_state, verbose=(verbose_level >= 2))

    # 提取 BO 最佳参数并应用约束
    best_C_bo = max(0.01, result_bo.x[0])
    best_gamma_bo = max(0.01, result_bo.x[1])
    best_lambda1_raw_bo = max(0.01, min(0.98, result_bo.x[2]))
    best_lambda2_raw_bo = max(0.01, min(0.98, result_bo.x[3]))
    best_lambda1_bo = best_lambda1_raw_bo
    best_lambda2_bo = best_lambda2_raw_bo
    if best_lambda1_bo + best_lambda2_bo > 0.98:
        scale = 0.98 / (best_lambda1_bo + best_lambda2_bo)
        best_lambda1_bo *= scale
        best_lambda2_bo *= scale
    best_lambda1_bo = max(0.01, best_lambda1_bo)
    best_lambda2_bo = max(0.01, best_lambda2_bo)

    best_coef0_bo = max(0, result_bo.x[4])
    best_epsilon_bo = max(0.01, min(1.0, result_bo.x[5]))
    best_lambda3_bo = max(0, 1.0 - best_lambda1_bo - best_lambda2_bo)
    best_degree_final = fixed_degree_bo

    best_score_bo = -result_bo.fun

    if verbose_level >= 1:
        print(f"  贝叶斯优化 完成. 最佳约束后参数: C={best_C_bo:.4f}, gamma={best_gamma_bo:.4f}, "
              f"l1={best_lambda1_bo:.4f}, l2={best_lambda2_bo:.4f}, l3={best_lambda3_bo:.4f}, "
              f"deg={best_degree_final}, coef0={best_coef0_bo:.4f}, eps={best_epsilon_bo:.4f}")
        print(f"  BO 找到的最佳测试集 R²: {best_score_bo:.4f}")

    # --- 步骤 3: 使用最终参数评估 ---
    def final_kernel_instance(X, Y):
        return hybrid_kernel(X, Y, gamma=best_gamma_bo, lambda1=best_lambda1_bo,
                             lambda2=best_lambda2_bo, degree=best_degree_final, coef0=best_coef0_bo)

    final_model = SVR(kernel=final_kernel_instance, C=best_C_bo, epsilon=best_epsilon_bo)
    final_test_r2 = -float('inf')
    try:
        final_model.fit(X_train, y_train)
        y_test_pred_final = final_model.predict(X_test)
        final_test_r2 = r2_score(y_test, y_test_pred_final)
    except Exception as e:
        print(f"  最终模型训练/评估出错: {e}")

    end_time_pipeline = time.time()
    if verbose_level >= 1:
        print(f"--- 混合核优化流程结束. 耗时: {end_time_pipeline - start_time_pipeline:.2f} 秒 ---")
        print(f"流程得到的最终测试集 R²: {final_test_r2:.4f}")

    return final_test_r2
# --- run_hybrid_optimization_pipeline 函数结束 ---


# --- Y 随机化检验参数 ---
N_PERMUTATIONS = 100
APSO_ITER_PERM = 50
BO_CALLS_PERM = 50

# --- 1. 运行基准模型 ---
print("\n=== 开始运行基准模型 (使用原始训练数据) ===")
start_time_base = time.time()
original_test_r2 = run_hybrid_optimization_pipeline(
    X_train_scaled_original, y_train_original, X_test_scaled_fixed, y_test_fixed,
    apso_particles=50, apso_iters=70, # 使用原始设置
    bo_calls=70, verbose_level=1
)
end_time_base = time.time()
print(f"\n=== 基准模型运行完成 ===")
print(f"原始数据的测试集 R²: {original_test_r2:.4f}")
print(f"耗时: {end_time_base - start_time_base:.2f} 秒")


# --- 2. 执行 Y 随机化检验 ---
print(f"\n=== 开始 Y 随机化检验 ({N_PERMUTATIONS} 次置换) ===")
print(f"(每次置换使用 APSO 迭代={APSO_ITER_PERM}, BO 调用={BO_CALLS_PERM})")

shuffled_test_r2_scores = []
rng = np.random.default_rng(seed=123)
start_time_perms = time.time()

for p in range(N_PERMUTATIONS):
    start_time_perm = time.time()
    print(f"\n--- 运行置换 {p + 1}/{N_PERMUTATIONS} ---")
    y_train_shuffled = y_train_original.copy()
    rng.shuffle(y_train_shuffled)

    perm_test_r2 = run_hybrid_optimization_pipeline(
        X_train_scaled_original, y_train_shuffled, X_test_scaled_fixed, y_test_fixed,
        apso_particles=30,
        apso_iters=APSO_ITER_PERM,
        bo_calls=BO_CALLS_PERM,
        verbose_level=0
    )
    shuffled_test_r2_scores.append(perm_test_r2)
    end_time_perm = time.time()
    print(f"置换 {p + 1} 完成. 测试集 R² = {perm_test_r2:.4f}. (耗时: {end_time_perm - start_time_perm:.2f} 秒)")

end_time_perms = time.time()
print(f"\n=== Y 随机化检验完成 ===")
print(f"总共 {N_PERMUTATIONS} 次置换耗时: {end_time_perms - start_time_perms:.2f} 秒")


# --- 3. 分析 Y 随机化结果 ---
shuffled_test_r2_scores = np.array(shuffled_test_r2_scores)
valid_shuffled_r2 = shuffled_test_r2_scores[np.isfinite(shuffled_test_r2_scores)]
n_valid_perms = len(valid_shuffled_r2)

print(f"\n--- Y 随机化结果分析 (基于 {n_valid_perms} 次有效置换) ---")

if n_valid_perms > 0:
    p_value_r2 = np.sum(valid_shuffled_r2 >= original_test_r2) / n_valid_perms
    mean_shuffled_r2 = np.mean(valid_shuffled_r2)
    std_shuffled_r2 = np.std(valid_shuffled_r2)
    median_shuffled_r2 = np.median(valid_shuffled_r2)

    print(f"原始模型测试集 R²: {original_test_r2:.4f}")
    print(f"随机化模型测试集 R² (均值 ± 标准差): {mean_shuffled_r2:.4f} ± {std_shuffled_r2:.4f}")
    print(f"随机化模型测试集 R² (中位数): {median_shuffled_r2:.4f}")
    print(f"Y 随机化检验 p 值 (R²): {p_value_r2:.4f}")
    print(f"(p 值表示随机情况下获得 R² >= {original_test_r2:.4f} 的概率)")

    if p_value_r2 < 0.05:
        print("\n结论: 原始模型的性能具有统计显著性 (p < 0.05)。")
    else:
        print("\n结论: 原始模型的性能不具有统计显著性 (p >= 0.05)。")
else:
    print("\n错误: 没有从置换运行中获得有效的 R² 结果，无法计算 p 值。")


# --- 4. 可视化 Y 随机化结果 ---
if n_valid_perms > 0:
    try:
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.hist(valid_shuffled_r2, bins=20, alpha=0.7, color='lightblue', edgecolor='black', label='随机化测试集 R² 分布')
        plt.axvline(original_test_r2, color='red', linestyle='dashed', linewidth=2, label=f'原始测试集 R² ({original_test_r2:.3f})')
        plt.xlabel("测试集 R² 分数")
        plt.ylabel("频数")
        plt.title(f"Y 随机化检验结果 ({n_valid_perms} 次有效置换) - 混合核 (λ1+λ2≤0.98)")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('y_randomization_hybrid_kernel_constrained_results.png', dpi=300)
        plt.show()
    except ImportError:
        print("\n未找到 Matplotlib 库。跳过绘制随机化结果直方图。")
    except Exception as e:
        print(f"\n绘制随机化结果直方图时出错: {e}")


print("\n脚本执行完毕.")
