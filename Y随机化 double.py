import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
# from sklearn.preprocessing import StandardScaler # Not used as data is assumed pre-scaled
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_squared_error # Added mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import random
import time # Import time for timing
from math import sqrt # Import sqrt

# --- 设置随机种子以确保结果可重复 ---
np.random.seed(40)
random.seed(40)

# --- 读取数据 ---
filename = '6result_guiyihua2.csv' # 确保文件名正确
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
test_indices = [5, 3, 11, 12, 13, 26, 27, 29, 40, 43, 46, 48, 49, 51]
if max(test_indices) >= len(X_all):
    print(f"错误：测试索引超出数据范围 (最大索引 {max(test_indices)}, 数据长度 {len(X_all)})。")
    exit()

# --- 分离训练集和测试集 (只执行一次) ---
X_test_fixed = X_all[test_indices]
y_test_fixed = y_all[test_indices]
X_train_original = np.delete(X_all, test_indices, axis=0)
y_train_original = np.delete(y_all, test_indices, axis=0)
print(f"训练集样本数: {len(X_train_original)}, 测试集样本数: {len(X_test_fixed)}")

# --- 数据归一化处理 (假设数据已预处理好) ---
# 如果数据未归一化，应在此处进行 StandardScaler 的 fit_transform(X_train) 和 transform(X_test)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train_original)
# X_test_scaled = scaler.transform(X_test_fixed)
# 由于原始代码直接使用，我们假设数据已归一化
X_train_scaled = X_train_original
X_test_scaled = X_test_fixed

# --- 自定义混合核函数 (放在全局，因为它会被多次调用) ---
def hybrid_kernel(X1, X2, gamma=1.0, lambda1=0.5, degree=3, coef0=1):
    """混合核函数: RBF + 多项式，增加稳定性处理"""
    lambda2 = 1.0 - lambda1 # 确保权重和为1

    # 确保输入是至少二维的
    if X1.ndim == 1: X1 = X1.reshape(1, -1)
    if X2.ndim == 1: X2 = X2.reshape(1, -1)

    # RBF 部分
    try:
        # 使用 cdist 计算欧氏距离平方
        sqdist = cdist(X1, X2, 'sqeuclidean')
        # 检查 gamma 是否过大导致下溢
        gamma = max(1e-9, gamma) # 防止 gamma 过小或为0
        rbf_part = np.exp(-gamma * sqdist)
        # 检查 NaN/Inf
        if np.any(np.isnan(rbf_part)) or np.any(np.isinf(rbf_part)):
            # print(f"警告: RBF 核计算出现 NaN/Inf (gamma={gamma}). 返回零矩阵.")
            rbf_part = np.zeros_like(sqdist) # 或者其他处理方式
    except Exception as e:
        # print(f"警告: RBF 核计算出错 (gamma={gamma}): {e}. 返回零矩阵.")
        rbf_part = np.zeros((X1.shape[0], X2.shape[0]))


    # 多项式部分
    try:
        degree = int(round(max(1, degree))) # 确保 degree 是正整数
        coef0 = max(0, coef0) # 确保 coef0 非负
        # 检查内积是否可能产生非常大或小的数
        poly_base = np.dot(X1, X2.T) + coef0
        # 检查底数是否为负且指数为非整数（虽然 degree 已转为 int，仍是好习惯）
        # 对非常大的底数和指数组合，结果可能溢出
        with np.errstate(over='raise', invalid='raise'): # 尝试捕捉溢出和无效操作
             try:
                 poly_part = np.power(poly_base, degree)
             except (FloatingPointError, ValueError): # 捕捉溢出或值错误
                 # print(f"警告: 多项式核计算溢出或值错误 (degree={degree}, coef0={coef0}). 尝试裁剪.")
                 # 尝试对 poly_base 进行裁剪或者返回一个大/小值
                 # 或者直接返回零矩阵，具体策略取决于问题
                 poly_part = np.zeros_like(poly_base)

        # 检查 NaN/Inf
        if np.any(np.isnan(poly_part)) or np.any(np.isinf(poly_part)):
            # print(f"警告: 多项式核计算出现 NaN/Inf (degree={degree}, coef0={coef0}). 返回零矩阵.")
            poly_part = np.zeros_like(poly_base)
    except Exception as e:
        # print(f"警告: 多项式核计算出错 (degree={degree}, coef0={coef0}): {e}. 返回零矩阵.")
        poly_part = np.zeros((X1.shape[0], X2.shape[0]))

    # 组合 - 再次检查NaN/Inf
    combined = lambda1 * rbf_part + lambda2 * poly_part
    if np.any(np.isnan(combined)) or np.any(np.isinf(combined)):
        # print("警告: 最终混合核计算出现 NaN/Inf. 返回零矩阵.")
        return np.zeros_like(combined)

    return combined


# --- 实现 APSO 算法 (保持不变，但增加 verbose 控制) ---
class APSO:
    def __init__(self, n_particles, dimensions, bounds, max_iter=100, verbose=True):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds # ([low1,...], [high1,...])
        self.max_iter = max_iter
        self.verbose = verbose

        # 初始化粒子位置和速度
        self.positions = np.zeros((n_particles, dimensions))
        self.velocities = np.zeros((n_particles, dimensions))
        self.best_positions = np.zeros((n_particles, dimensions))
        self.best_scores = np.ones(n_particles) * float('inf') # 适应度，越小越好
        self.global_best_position = np.zeros(dimensions)
        self.global_best_score = float('inf')

        # 初始化粒子位置和速度
        for i in range(dimensions):
            lower_bound = bounds[0][i]
            upper_bound = bounds[1][i]
            if lower_bound >= upper_bound:
                 upper_bound = lower_bound + 1e-6 # 保证上界大于下界
            self.positions[:, i] = np.random.uniform(lower_bound, upper_bound, n_particles)
            vel_range = upper_bound - lower_bound
            self.velocities[:, i] = np.random.uniform(-1, 1, n_particles) * max(vel_range, 1e-6) * 0.1
        self.best_positions = self.positions.copy()

    def update_parameters(self, iter_num, f_avg, f_min, f):
        """更新APSO的参数（惯性权重和学习因子）"""
        w_max, w_min = 0.9, 0.4
        c1_max, c1_min = 2.5, 0.5
        c2_max, c2_min = 2.5, 0.5

        # 计算进化因子 (避免除零)
        safe_f_avg = f_avg
        safe_f_min = f_min
        if abs(f_avg - f_min) < 1e-10:
            safe_f_avg = f_min + 1e-10

        f_values = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            denom = safe_f_avg - safe_f_min
            if abs(denom) < 1e-10:
                f_values[i] = 0.0 if abs(f[i] - safe_f_min) < 1e-10 else 1.0
            else:
                # 确保 f_values 非负，并且在合理范围内
                f_values[i] = np.clip((f[i] - safe_f_min) / denom, 0, 10) # 限制最大值，防止极端情况

        # 根据进化因子调整参数
        w_values = np.zeros(self.n_particles)
        c1_values = np.zeros(self.n_particles)
        c2_values = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            w_values[i] = w_max - (w_max - w_min) * iter_num / self.max_iter
            f_val_clipped = np.clip(f_values[i], 0, 1) # 只使用 [0, 1] 范围进行调整
            if f_val_clipped >= 0.5:  # 探索
                c1_values[i] = c1_min + (c1_max - c1_min) * (f_val_clipped - 0.5) * 2
                c2_values[i] = c2_max - (c2_max - c2_min) * (f_val_clipped - 0.5) * 2
            else:  # 开发
                c1_values[i] = c1_max - (c1_max - c1_min) * f_val_clipped * 2
                c2_values[i] = c2_min + (c2_max - c2_min) * f_val_clipped * 2
            c1_values[i] = np.clip(c1_values[i], c1_min, c1_max)
            c2_values[i] = np.clip(c2_values[i], c2_min, c2_max)
        return w_values, c1_values, c2_values

    def optimize(self, objective_function):
        """执行APSO优化"""
        # 初始评估，处理可能的错误
        try:
            scores = objective_function(self.positions)
            # 检查是否有非有限值
            if not np.all(np.isfinite(scores)):
                print("警告: APSO 初始评估包含非有限值。将替换为极大值。")
                scores[~np.isfinite(scores)] = float('inf')
        except Exception as e:
            print(f"错误: APSO 初始评估失败: {e}")
            print("      所有粒子得分将设为极大值。")
            scores = np.ones(self.n_particles) * float('inf')

        # 更新个体和全局最优
        for i in range(self.n_particles):
            if scores[i] < self.best_scores[i]:
                self.best_scores[i] = scores[i]
                self.best_positions[i] = self.positions[i].copy()
            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.positions[i].copy()

        # 迭代优化
        for iter_num in range(self.max_iter):
            valid_scores = scores[np.isfinite(scores)]
            if len(valid_scores) == 0 or self.global_best_score == float('inf'): # 如果所有得分无效或从未找到有效解
                 if self.verbose:
                     print(f"  APSO 警告: 第 {iter_num+1} 次迭代无有效得分，跳过参数更新。")
                 # 可以选择重新初始化部分粒子或采取其他策略
                 # 这里简单跳过更新，但保持粒子移动
                 w_values = np.ones(self.n_particles) * 0.7
                 c1_values = np.ones(self.n_particles) * 1.5
                 c2_values = np.ones(self.n_particles) * 1.5
            else:
                f_avg = np.mean(valid_scores)
                f_min = np.min(valid_scores) # 确保 f_min 来自有效得分
                # 使用全局最佳得分来更新 f_min，这可能更稳定
                f_min = min(f_min, self.global_best_score)
                w_values, c1_values, c2_values = self.update_parameters(iter_num, f_avg, f_min, scores)

            # 更新速度和位置
            for i in range(self.n_particles):
                r1, r2 = np.random.random(self.dimensions), np.random.random(self.dimensions)
                # 确保全局最优位置是有效的
                if self.global_best_score == float('inf'):
                    # 如果还没有全局最优，让粒子随机游走或使用个体最优
                    social = c1_values[i] * r2 * (self.best_positions[i] - self.positions[i]) # 模仿认知部分
                else:
                    social = c2_values[i] * r2 * (self.global_best_position - self.positions[i])
                cognitive = c1_values[i] * r1 * (self.best_positions[i] - self.positions[i])
                self.velocities[i] = w_values[i] * self.velocities[i] + cognitive + social

                # 限制速度
                for j in range(self.dimensions):
                    vel_range = self.bounds[1][j] - self.bounds[0][j]
                    max_velocity = 0.2 * max(vel_range, 1e-6)
                    self.velocities[i, j] = np.clip(self.velocities[i, j], -max_velocity, max_velocity)
                # 更新位置
                self.positions[i] += self.velocities[i]
                # 边界处理
                for j in range(self.dimensions):
                    self.positions[i, j] = np.clip(self.positions[i, j], self.bounds[0][j], self.bounds[1][j])

            # 评估新位置
            try:
                scores = objective_function(self.positions)
                if not np.all(np.isfinite(scores)):
                    # print("警告: APSO 评估包含非有限值。将替换为极大值。")
                    scores[~np.isfinite(scores)] = float('inf')
            except Exception as e:
                 print(f"错误: APSO 评估失败 (迭代 {iter_num+1}): {e}. 得分设为极大值。")
                 scores = np.ones(self.n_particles) * float('inf')


            # 更新个体和全局最优
            for i in range(self.n_particles):
                if scores[i] < self.best_scores[i]:
                    self.best_scores[i] = scores[i]
                    self.best_positions[i] = self.positions[i].copy()
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i].copy()

            # 打印进度
            if self.verbose and ((iter_num + 1) % 10 == 0 or iter_num == 0): # 减少打印频率
                current_best_display = -self.global_best_score if self.global_best_score != float('inf') else float('-inf')
                print(f"  APSO 迭代 {iter_num + 1}/{self.max_iter}, 当前最佳得分 (负目标函数值): {current_best_display:.4f}")

        # 返回最终结果（得分取反，位置）
        final_best_score = -self.global_best_score if self.global_best_score != float('inf') else -float('inf')
        # 确保返回的位置是有效的
        if self.global_best_score == float('inf'):
             print("警告: APSO未能找到有效的全局最优解。返回初始估计或零向量。")
             # 可以返回 bounds 的中心或者一个随机位置
             return -float('inf'), (np.array(self.bounds[0]) + np.array(self.bounds[1])) / 2

        return final_best_score, self.global_best_position.copy()
# --- APSO 类结束 ---


# --- 封装优化流程的函数 ---
def run_hybrid_optimization_pipeline(X_train, y_train, X_test, y_test,
                                     apso_particles=50, apso_iters=70,
                                     bo_calls=70, verbose_level=1):
    """
    执行完整的混合核SVR的APSO-BO优化流程并返回测试集R²。

    Args:
        X_train (np.ndarray): 训练特征 (假设已缩放)
        y_train (np.ndarray): 训练目标
        X_test (np.ndarray): 测试特征 (假设已缩放)
        y_test (np.ndarray): 测试目标
        apso_particles (int): APSO粒子数
        apso_iters (int): APSO迭代次数
        bo_calls (int): 贝叶斯优化调用次数
        verbose_level (int): 控制输出详细程度 (0: 无输出, 1: 基本输出, 2: 详细输出)

    Returns:
        float: 在测试集上的 R² 分数。如果优化或评估失败，返回 -inf。
    """
    start_time_pipeline = time.time()
    if verbose_level >= 1:
        print("--- 开始执行混合核优化流程 ---")

    # --- 步骤 1: 使用APSO进行粗略搜索 ---
    # APSO 目标函数 (内部定义，访问外部 X/y)
    def apso_objective_inner(params_array):
        results = []
        for param_set in params_array:
            C = max(1e-5, param_set[0])
            gamma = max(1e-5, param_set[1])
            lambda1 = max(1e-5, min(1.0 - 1e-5, param_set[2])) # 确保 lambda 在 (0, 1) 开区间附近
            degree = int(round(max(1, param_set[3]))) # 至少为1
            coef0 = max(0, param_set[4])
            epsilon = max(1e-5, min(1.0, param_set[5])) # 限制 epsilon
            score = -float('inf') # 默认失败得分

            try:
                # 定义当前参数的核函数
                def current_kernel(X, Y):
                    return hybrid_kernel(X, Y, gamma=gamma, lambda1=lambda1,
                                         degree=degree, coef0=coef0)

                model = SVR(kernel=current_kernel, C=C, epsilon=epsilon)
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                score = r2_score(y_test, y_test_pred)
                # 处理 R2 可能为 NaN 或 Inf 的情况
                if not np.isfinite(score):
                    score = -float('inf')
            except MemoryError:
                 if verbose_level >=2:
                     print(f"  APSO 内部内存错误: 参数 C={C:.1e}, gam={gamma:.1e}, lam={lambda1:.1e}, deg={degree}, coef={coef0:.1e}, eps={epsilon:.1e}.")
                 score = -float('inf') # 内存错误通常意味着参数组合不佳
            except Exception as e:
                if verbose_level >= 2:
                    # 减少打印频率或只打印关键错误
                    # if 'Input contains NaN' in str(e) or 'Input contains infinity' in str(e):
                    #      print(f"  APSO 内部数据错误: {e}") # 这通常指示核函数问题
                    # else:
                    #      print(f"  APSO 内部错误: 参数 C={C:.1e}, gam={gamma:.1e}, lam={lambda1:.1e}, deg={degree}, coef={coef0:.1e}, eps={epsilon:.1e}. 错误: {e}")
                    pass # 暂时抑制一些错误打印，避免刷屏
                score = -float('inf')

            results.append(-score) # APSO 最小化负 R²
        return np.array(results)

    # APSO 参数范围 (与全局定义一致)
    bounds_apso = ([0.01, 0.01, 0.01, 2, 0, 0.01], [100.0, 10.0, 0.99, 4, 5, 1.0])
    apso_optimizer = APSO(n_particles=apso_particles, dimensions=6, bounds=bounds_apso,
                          max_iter=apso_iters, verbose=(verbose_level >= 2))

    if verbose_level >= 1:
        print("  开始APSO优化...")
    best_score_apso, best_pos_apso = apso_optimizer.optimize(apso_objective_inner)

    # 提取并约束 APSO 参数
    best_C_apso = max(1e-5, best_pos_apso[0])
    best_gamma_apso = max(1e-5, best_pos_apso[1])
    best_lambda1_apso = max(1e-5, min(1.0 - 1e-5, best_pos_apso[2]))
    best_degree_apso = int(round(max(1, best_pos_apso[3])))
    best_coef0_apso = max(0, best_pos_apso[4])
    best_epsilon_apso = max(1e-5, min(1.0, best_pos_apso[5]))

    if verbose_level >= 1:
        print(f"  APSO 完成. 最佳参数估计: C={best_C_apso:.3f}, gam={best_gamma_apso:.3f}, lam={best_lambda1_apso:.3f}, deg={best_degree_apso}, coef={best_coef0_apso:.3f}, eps={best_epsilon_apso:.3f}")
        print(f"  APSO 找到的最佳测试集 R² (近似): {best_score_apso:.4f}") # best_score_apso 是 R²

    # 如果 APSO 失败，提前退出
    if best_score_apso <= -float('inf'):
        print("  APSO 未能找到有效解，优化流程终止。")
        return -float('inf')

    # --- 步骤 2: 使用贝叶斯优化进行精细调优 ---
    # 固定 degree
    best_degree_fixed = best_degree_apso

    # 定义 BO 空间 (围绕 APSO 结果，确保下界 < 上界)
    c_low = max(1e-5, best_C_apso * 0.8)
    c_high = max(c_low + 1e-4, min(100.0, best_C_apso * 1.2))
    g_low = max(1e-5, best_gamma_apso * 0.8)
    g_high = max(g_low + 1e-4, min(10.0, best_gamma_apso * 1.2))
    l_low = max(1e-5, best_lambda1_apso * 0.8)
    l_high = min(1.0 - 1e-5, best_lambda1_apso * 1.2)
    if l_low >= l_high: l_high = min(1.0 - 1e-5, l_low + 0.01) # 确保上界有效
    co_low = max(0, best_coef0_apso * 0.8) # coef0 可以为 0
    co_high = max(co_low + 1e-4, min(5.0, best_coef0_apso * 1.2))
    ep_low = max(1e-5, best_epsilon_apso * 0.8)
    ep_high = max(ep_low + 1e-4, min(1.0, best_epsilon_apso * 1.2))

    space_bo = [
        Real(c_low, c_high, name='C', prior='log-uniform'),
        Real(g_low, g_high, name='gamma', prior='log-uniform'),
        Real(l_low, l_high, name='lambda1', prior='uniform'), # lambda 在 (0,1) 之间，用 uniform
        Real(co_low, co_high, name='coef0', prior='uniform'),
        Real(ep_low, ep_high, name='epsilon', prior='log-uniform')
    ]

    # BO 目标函数 (内部定义)
    @use_named_args(space_bo)
    def bo_objective_inner(C, gamma, lambda1, coef0, epsilon):
        score = -float('inf')
        try:
            # 约束参数
            C = max(1e-5, C)
            gamma = max(1e-5, gamma)
            lambda1 = max(1e-5, min(1.0 - 1e-5, lambda1))
            coef0 = max(0, coef0)
            epsilon = max(1e-5, min(1.0, epsilon))

            def current_kernel_bo(X, Y):
                return hybrid_kernel(X, Y, gamma=gamma, lambda1=lambda1,
                                     degree=best_degree_fixed, coef0=coef0)

            model = SVR(kernel=current_kernel_bo, C=C, epsilon=epsilon)
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            score = r2_score(y_test, y_test_pred)
            if not np.isfinite(score): score = -float('inf')
            return -score # BO 最小化负 R²
        except MemoryError:
             if verbose_level >=2:
                  print(f"  BO 内部内存错误.")
             return 1e6 # 返回大的惩罚值
        except Exception as e:
            if verbose_level >= 2:
                 # print(f"  BO 内部错误: C={C:.1e}, gam={gamma:.1e}, lam={lambda1:.1e}, coef={coef0:.1e}, eps={epsilon:.1e}. 错误: {e}")
                 pass
            return 1e6 # 返回大的惩罚值

    if verbose_level >= 1:
        print("\n  开始贝叶斯优化...")
    # 运行 BO
    bo_random_state = int(time.time()) % 1000 + 42 # 结合时间和固定种子
    # 增加 n_restarts_optimizer 可能有助于避免局部最优
    # acq_func='EI' (Expected Improvement) 或 'gp_hedge'
    result_bo = gp_minimize(bo_objective_inner, space_bo, n_calls=bo_calls,
                             random_state=bo_random_state, verbose=(verbose_level >= 2),
                             n_restarts_optimizer=10, # 增加重启次数
                             noise=1e-6) # 可以添加少量噪声项

    # 提取并约束 BO 参数
    best_C_bo = max(1e-5, result_bo.x[0])
    best_gamma_bo = max(1e-5, result_bo.x[1])
    best_lambda1_bo = max(1e-5, min(1.0 - 1e-5, result_bo.x[2]))
    best_coef0_bo = max(0, result_bo.x[3])
    best_epsilon_bo = max(1e-5, min(1.0, result_bo.x[4]))
    best_score_bo = -result_bo.fun # BO 找到的最佳测试集 R² (可能为 -inf)

    if verbose_level >= 1:
        print(f"  贝叶斯优化 完成. 最佳参数: C={best_C_bo:.3f}, gam={best_gamma_bo:.3f}, lam={best_lambda1_bo:.3f}, deg={best_degree_fixed}, coef={best_coef0_bo:.3f}, eps={best_epsilon_bo:.3f}")
        print(f"  BO 找到的最佳测试集 R²: {best_score_bo:.4f}")

    # --- 步骤 3: 使用最终参数评估 ---
    final_test_r2 = -float('inf') # 默认失败
    if best_score_bo > -float('inf'): # 仅当 BO 成功时才进行最终评估
        try:
            def final_kernel_inner(X, Y):
                 return hybrid_kernel(X, Y, gamma=best_gamma_bo, lambda1=best_lambda1_bo,
                                      degree=best_degree_fixed, coef0=best_coef0_bo)

            final_model = SVR(kernel=final_kernel_inner, C=best_C_bo, epsilon=best_epsilon_bo)
            final_model.fit(X_train, y_train)
            y_test_pred_final = final_model.predict(X_test)
            final_test_r2 = r2_score(y_test, y_test_pred_final)
            if not np.isfinite(final_test_r2): final_test_r2 = -float('inf')

            # (可选) 计算 RMSE
            # final_test_rmse = sqrt(mean_squared_error(y_test, y_test_pred_final)) if np.isfinite(final_test_r2) else float('inf')

        except Exception as e:
            print(f"  最终模型训练/评估出错: {e}")
            final_test_r2 = -float('inf')
    else:
         print("  BO 未找到有效解，跳过最终模型评估。")


    end_time_pipeline = time.time()
    if verbose_level >= 1:
        print(f"--- 混合核优化流程结束. 耗时: {end_time_pipeline - start_time_pipeline:.2f} 秒 ---")
        print(f"流程得到的最终测试集 R²: {final_test_r2:.4f}")

    return final_test_r2
# --- run_hybrid_optimization_pipeline 函数结束 ---


# --- Y 随机化检验参数 ---
N_PERMUTATIONS = 100  # 置换次数 (建议至少 100)
# 在随机化时使用的迭代次数 (可以减少以加速)
APSO_ITER_PERM = 50
BO_CALLS_PERM = 50

# --- 1. 在原始数据上运行基准优化流程 ---
print("\n=== 开始运行基准模型 (使用原始训练数据) ===")
start_time_base = time.time()
# 调用核心函数，使用原始数据和标准迭代次数
original_test_r2 = run_hybrid_optimization_pipeline(
    X_train_scaled, y_train_original, X_test_scaled, y_test_fixed,
    apso_particles=50, apso_iters=70, # 使用原始设置
    bo_calls=70, verbose_level=1 # 基本输出
)
end_time_base = time.time()
print(f"\n=== 基准模型运行完成 ===")
print(f"原始数据的测试集 R²: {original_test_r2:.4f}")
print(f"耗时: {end_time_base - start_time_base:.2f} 秒")

# --- 2. 执行 Y 随机化检验 ---
print(f"\n=== 开始 Y 随机化检验 ({N_PERMUTATIONS} 次置换) ===")
print(f"(每次置换使用 APSO 迭代={APSO_ITER_PERM}, BO 调用={BO_CALLS_PERM})")

shuffled_test_r2_scores = [] # 存储每次置换的测试集 R²

# 创建用于洗牌的随机数生成器
rng = np.random.default_rng(seed=123) # 固定洗牌种子

start_time_perms = time.time()
for p in range(N_PERMUTATIONS):
    start_time_perm = time.time()
    print(f"\n--- 运行置换 {p + 1}/{N_PERMUTATIONS} ---")

    # 创建 y_train 的副本并洗牌
    y_train_shuffled = y_train_original.copy()
    rng.shuffle(y_train_shuffled) # 原地洗牌

    # 使用洗牌后的 y_train 运行优化流程 (减少迭代次数, 关闭内部详细输出)
    perm_test_r2 = run_hybrid_optimization_pipeline(
        X_train_scaled, y_train_shuffled, X_test_scaled, y_test_fixed,
        apso_particles=40, # 可适当减少粒子数
        apso_iters=APSO_ITER_PERM,
        bo_calls=BO_CALLS_PERM,
        verbose_level=0 # 关闭内部输出
    )

    shuffled_test_r2_scores.append(perm_test_r2)
    end_time_perm = time.time()
    # 打印时处理可能的 -inf
    perm_r2_display = f"{perm_test_r2:.4f}" if np.isfinite(perm_test_r2) else "-inf"
    print(f"置换 {p + 1} 完成. 测试集 R² = {perm_r2_display}. (耗时: {end_time_perm - start_time_perm:.2f} 秒)")

end_time_perms = time.time()
print(f"\n=== Y 随机化检验完成 ===")
print(f"总共 {N_PERMUTATIONS} 次置换耗时: {end_time_perms - start_time_perms:.2f} 秒")

# --- 3. 分析 Y 随机化结果 ---
shuffled_test_r2_scores = np.array(shuffled_test_r2_scores)
# 过滤掉无效 R² 值 (-inf)
valid_shuffled_r2 = shuffled_test_r2_scores[np.isfinite(shuffled_test_r2_scores)]
n_valid_perms = len(valid_shuffled_r2)
n_failed_perms = N_PERMUTATIONS - n_valid_perms

print(f"\n--- Y 随机化结果分析 (基于 {n_valid_perms} 次有效置换, {n_failed_perms} 次失败) ---")

# 只有在原始 R2 有效且至少有一次有效置换时才计算 p 值
if np.isfinite(original_test_r2) and n_valid_perms > 0:
    # 计算 p 值: 置换 R² >= 原始 R² 的比例
    p_value_r2 = np.sum(valid_shuffled_r2 >= original_test_r2) / n_valid_perms

    # 计算置换 R² 的统计信息
    mean_shuffled_r2 = np.mean(valid_shuffled_r2)
    std_shuffled_r2 = np.std(valid_shuffled_r2)
    median_shuffled_r2 = np.median(valid_shuffled_r2)

    print(f"原始模型测试集 R²: {original_test_r2:.4f}")
    print(f"随机化模型测试集 R² (均值 ± 标准差): {mean_shuffled_r2:.4f} ± {std_shuffled_r2:.4f}")
    print(f"随机化模型测试集 R² (中位数): {median_shuffled_r2:.4f}")
    print(f"Y 随机化检验 p 值 (R²): {p_value_r2:.4f}")
    print(f"(p 值表示随机情况下获得 R² >= {original_test_r2:.4f} 的概率)")

    # 结论
    if p_value_r2 < 0.05:
        print("\n结论: 原始模型的性能具有统计显著性 (p < 0.05)。")
        print("      模型可能学习到了真实的 X-y 关系，而不仅仅是拟合了噪声或偶然性。")
    else:
        print("\n结论: 原始模型的性能不具有统计显著性 (p >= 0.05)。")
        print("      需要谨慎对待模型结果，可能存在过拟合或模型未能有效学习到 X-y 关系。")
    if n_failed_perms > N_PERMUTATIONS * 0.1: # 如果失败次数较多
        print(f"警告: 有 {n_failed_perms} ({n_failed_perms/N_PERMUTATIONS:.1%}) 次置换未能成功完成优化或评估，这可能影响 p 值的可靠性。")
        print("      建议检查优化过程中的错误信息或参数设置。")

elif not np.isfinite(original_test_r2):
     print("\n错误: 原始基准模型未能获得有效的 R² 分数。无法进行 Y 随机化比较。")
else: # n_valid_perms == 0
    print("\n错误: 所有置换运行均未能获得有效的 R² 结果。无法计算 p 值。")
    print("      请检查优化流程在处理随机化数据时是否频繁失败。")


# --- 4. 可视化 Y 随机化结果 ---
if n_valid_perms > 0 and np.isfinite(original_test_r2):
    try:
        plt.figure(figsize=(8, 6))
        # 设置 matplotlib 支持中文显示
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
            plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
        except Exception as font_e:
             print(f"设置中文字体失败: {font_e}. 图表可能无法正确显示中文。")

        plt.hist(valid_shuffled_r2, bins=20, alpha=0.7, color='lightblue', edgecolor='black', label=f'随机化测试集 R² ({n_valid_perms} 次有效值)')
        plt.axvline(original_test_r2, color='red', linestyle='dashed', linewidth=2, label=f'原始测试集 R² ({original_test_r2:.3f})')

        # 添加均值和中位数的线（可选）
        plt.axvline(mean_shuffled_r2, color='orange', linestyle='dotted', linewidth=1.5, label=f'随机化均值 ({mean_shuffled_r2:.3f})')
        # plt.axvline(median_shuffled_r2, color='purple', linestyle='dotted', linewidth=1.5, label=f'随机化中位数 ({median_shuffled_r2:.3f})')

        plt.xlabel("测试集 R² 分数")
        plt.ylabel("频数")
        plt.title(f"Y 随机化检验结果 ({N_PERMUTATIONS} 次置换)")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('y_randomization_hybrid_results.png', dpi=300) # 保存图像
        plt.show()
    except ImportError:
        print("\n未找到 Matplotlib 库。跳过绘制随机化结果直方图。")
    except Exception as e:
        print(f"\n绘制随机化结果直方图时出错: {e}")
else:
    print("\n由于原始 R² 无效或无有效置换结果，跳过绘制直方图。")


print("\n脚本执行完毕.")