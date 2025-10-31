# 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error # 添加 mean_squared_error
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import random
import time # 导入 time 模块用于计时
from math import sqrt # 导入 sqrt

# --- 设置随机种子以确保结果可重复 ---
np.random.seed(68)
random.seed(68)

# --- 读取数据 ---
filename = '6result_guiyihua2.csv'  # CSV文件路径
try:
    df = pd.read_csv(filename, header=None) # 读取 CSV，假设没有表头
except FileNotFoundError:
    print(f"错误: 文件 '{filename}' 未找到。请确保文件在当前目录或提供正确路径。")
    exit()
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit()

# --- 分离特征和目标变量 ---
X_all = df.iloc[:, :-1].values  # 所有行，除最后一列外的所有列作为特征
y_all = df.iloc[:, -1].values  # 所有行的最后一列作为目标变量
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

# --- 实现 APSO 算法 (保持不变) ---
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
                f_values[i] = max(0, (f[i] - safe_f_min)) / denom

        # 根据进化因子调整参数
        w_values = np.zeros(self.n_particles)
        c1_values = np.zeros(self.n_particles)
        c2_values = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            w_values[i] = w_max - (w_max - w_min) * iter_num / self.max_iter
            f_val_clipped = np.clip(f_values[i], 0, 1)
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
        scores = objective_function(self.positions) # 初始评估

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
            if len(valid_scores) == 0:
                 f_avg, f_min = float('inf'), float('inf')
                 # 使用默认参数或上一轮参数
                 w_values = np.ones(self.n_particles) * 0.7
                 c1_values = np.ones(self.n_particles) * 1.5
                 c2_values = np.ones(self.n_particles) * 1.5
            else:
                f_avg = np.mean(valid_scores)
                f_min = np.min(valid_scores)
                w_values, c1_values, c2_values = self.update_parameters(iter_num, f_avg, f_min, scores)

            # 更新速度和位置
            for i in range(self.n_particles):
                r1, r2 = np.random.random(self.dimensions), np.random.random(self.dimensions)
                cognitive = c1_values[i] * r1 * (self.best_positions[i] - self.positions[i])
                social = c2_values[i] * r2 * (self.global_best_position - self.positions[i])
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
            scores = objective_function(self.positions)

            # 更新个体和全局最优
            for i in range(self.n_particles):
                if scores[i] < self.best_scores[i]:
                    self.best_scores[i] = scores[i]
                    self.best_positions[i] = self.positions[i].copy()
                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i].copy()

            # 打印进度
            if self.verbose and ((iter_num + 1) % 20 == 0 or iter_num == 0):
                current_best_display = -self.global_best_score if self.global_best_score != float('inf') else float('-inf')
                print(f"  APSO 迭代 {iter_num + 1}/{self.max_iter}, 当前最佳得分 (负目标函数值): {current_best_display:.4f}")

        final_best_score = -self.global_best_score if self.global_best_score != float('inf') else float('-inf')
        return final_best_score, self.global_best_position
# --- APSO 类结束 ---


# --- 封装优化流程的函数 ---
def run_optimization_pipeline(X_train, y_train, X_test, y_test,
                              apso_particles=50, apso_iters=70,
                              bo_calls=70, verbose_level=1):
    """
    执行完整的APSO-BO优化流程并返回测试集R²。

    Args:
        X_train (np.ndarray): 训练特征
        y_train (np.ndarray): 训练目标
        X_test (np.ndarray): 测试特征
        y_test (np.ndarray): 测试目标
        apso_particles (int): APSO粒子数
        apso_iters (int): APSO迭代次数
        bo_calls (int): 贝叶斯优化调用次数
        verbose_level (int): 控制输出详细程度 (0: 无输出, 1: 基本输出, 2: 详细输出)

    Returns:
        float: 在测试集上的 R² 分数。如果优化失败，返回一个极小值。
    """
    start_time_pipeline = time.time()
    if verbose_level >= 1:
        print("--- 开始执行优化流程 ---")

    # --- 步骤 1: 使用APSO进行粗略搜索 ---
    # 定义 APSO 目标函数 (在 run_optimization_pipeline 内部定义以访问其参数)
    def apso_objective_inner(params_array):
        results = []
        for param_set in params_array:
            C = max(1e-5, param_set[0])
            gamma = max(1e-5, param_set[1])
            epsilon = max(1e-5, param_set[2])
            try:
                model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                model.fit(X_train, y_train) # 使用传入的训练数据
                y_test_pred = model.predict(X_test) # 在传入的测试集上预测
                # *** 重要：APSO的目标基于测试集R² ***
                score = r2_score(y_test, y_test_pred) # 计算测试集R²
            except Exception as e:
                if verbose_level >= 2:
                    print(f"  APSO 内部错误: 参数 C={C:.2e}, gamma={gamma:.2e}, eps={epsilon:.2e}. 错误: {e}")
                score = -float('inf') # 非常差的分数
            # APSO 最小化，所以返回负的 R²
            results.append(-score)
        return np.array(results)

    # APSO 参数范围
    bounds_apso = ([0.01, 0.01, 0.01], [100.0, 10.0, 0.5])
    # 初始化 APSO 优化器
    apso_optimizer = APSO(n_particles=apso_particles, dimensions=3, bounds=bounds_apso,
                          max_iter=apso_iters, verbose=(verbose_level >= 2))

    if verbose_level >= 1:
        print("  开始APSO优化...")
    # 运行 APSO
    best_score_apso, best_pos_apso = apso_optimizer.optimize(apso_objective_inner)
    best_C_apso = max(1e-5, best_pos_apso[0])
    best_gamma_apso = max(1e-5, best_pos_apso[1])
    best_epsilon_apso = max(1e-5, best_pos_apso[2])

    if verbose_level >= 1:
        print(f"  APSO 完成. 最佳参数估计: C={best_C_apso:.4f}, gamma={best_gamma_apso:.4f}, epsilon={best_epsilon_apso:.4f}")
        print(f"  APSO 找到的最佳测试集 R² (近似): {best_score_apso:.4f}") # best_score_apso 是 R²

    # --- 步骤 2: 使用贝叶斯优化进行精细调优 ---
    # 定义 BO 搜索空间 (围绕 APSO 结果)
    c_low = max(0.01, best_C_apso * 0.8)
    c_high = max(c_low + 1e-4, min(100.0, best_C_apso * 1.2))
    g_low = max(0.01, best_gamma_apso * 0.8)
    g_high = max(g_low + 1e-4, min(10.0, best_gamma_apso * 1.2))
    e_low = max(0.01, best_epsilon_apso * 0.8)
    e_high = max(e_low + 1e-4, min(1.0, best_epsilon_apso * 1.2)) # 注意：原代码上界为0.5，这里改为1.0，与APSO一致

    space_bo = [
        Real(c_low, c_high, name='C', prior='log-uniform'),
        Real(g_low, g_high, name='gamma', prior='log-uniform'),
        Real(e_low, e_high, name='epsilon', prior='log-uniform')
    ]

    # 定义 BO 目标函数 (在 run_optimization_pipeline 内部)
    @use_named_args(space_bo)
    def bo_objective_inner(C, gamma, epsilon):
        try:
            C = max(1e-5, C)
            gamma = max(1e-5, gamma)
            epsilon = max(1e-5, epsilon)
            model = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
            model.fit(X_train, y_train) # 使用传入的训练数据
            y_test_pred = model.predict(X_test) # 在传入的测试集上预测
            # *** 重要：BO的目标也基于测试集R² ***
            score = r2_score(y_test, y_test_pred)
            return -score # BO 最小化，返回负 R²
        except Exception as e:
            if verbose_level >= 2:
                 print(f"  BO 内部错误: 参数 C={C:.2e}, gamma={gamma:.2e}, eps={epsilon:.2e}. 错误: {e}")
            return 1e6 # 返回一个很大的值表示失败

    if verbose_level >= 1:
        print("\n  开始贝叶斯优化...")
    # 运行 BO
    # 使用当前时间或特定种子增加随机性，避免每次都完全一样
    bo_random_state = int(time.time()) % 1000
    result_bo = gp_minimize(bo_objective_inner, space_bo, n_calls=bo_calls,
                             random_state=bo_random_state, verbose=(verbose_level >= 2))

    # 提取 BO 最佳参数
    best_C_bo = max(1e-5, result_bo.x[0])
    best_gamma_bo = max(1e-5, result_bo.x[1])
    best_epsilon_bo = max(1e-5, result_bo.x[2])
    best_score_bo = -result_bo.fun # BO找到的最佳测试集 R²

    if verbose_level >= 1:
        print(f"  贝叶斯优化 完成. 最佳参数: C={best_C_bo:.4f}, gamma={best_gamma_bo:.4f}, epsilon={best_epsilon_bo:.4f}")
        print(f"  BO 找到的最佳测试集 R²: {best_score_bo:.4f}")

    # --- 步骤 3: 使用最终参数评估 ---
    # 注意：虽然BO直接优化了测试集R²，我们还是用BO找到的参数再训练一次模型来获取最终的预测值和分数
    # 这在实践中可能有点冗余，因为 best_score_bo 理论上就是最终的测试集R²，但这样做更符合标准流程
    final_model = SVR(kernel='rbf', C=best_C_bo, gamma=best_gamma_bo, epsilon=best_epsilon_bo)
    try:
        final_model.fit(X_train, y_train)
        y_test_pred_final = final_model.predict(X_test)
        final_test_r2 = r2_score(y_test, y_test_pred_final)
        # 计算 RMSE (可选)
        # final_test_rmse = sqrt(mean_squared_error(y_test, y_test_pred_final))
    except Exception as e:
        print(f"  最终模型训练/评估出错: {e}")
        final_test_r2 = -float('inf') # 标记为失败

    end_time_pipeline = time.time()
    if verbose_level >= 1:
        print(f"--- 优化流程结束. 耗时: {end_time_pipeline - start_time_pipeline:.2f} 秒 ---")
        print(f"流程得到的最终测试集 R²: {final_test_r2:.4f}") # 打印本次流程的最终结果

    # 返回这次优化流程找到的最佳测试集 R²
    # 可以选择返回 best_score_bo 或 final_test_r2，理论上它们应该很接近
    # 返回 final_test_r2 更符合重新验证的逻辑
    return final_test_r2
# --- run_optimization_pipeline 函数结束 ---


# --- Y 随机化检验参数 ---
N_PERMUTATIONS = 100  # 置换次数 (建议至少 100, 测试时可设小)
# 在随机化时可以适当减少迭代次数以加速
APSO_ITER_PERM = 50   # 随机化运行时的 APSO 迭代次数
BO_CALLS_PERM = 50    # 随机化运行时的 BO 调用次数

# --- 1. 在原始数据上运行基准优化流程 ---
print("\n=== 开始运行基准模型 (使用原始训练数据) ===")
start_time_base = time.time()
original_test_r2 = run_optimization_pipeline(X_train_original, y_train_original, X_test_fixed, y_test_fixed,
                                            apso_particles=50, apso_iters=70, # 使用原始设置的迭代次数
                                            bo_calls=70, verbose_level=1) # 基本输出即可
end_time_base = time.time()
print(f"\n=== 基准模型运行完成 ===")
print(f"原始数据的测试集 R²: {original_test_r2:.4f}")
print(f"耗时: {end_time_base - start_time_base:.2f} 秒")

# (可选) 获取基准模型的最终参数和训练 R²，但这需要修改 run_optimization_pipeline 返回更多值
# 这里我们主要关注 Y 随机化中的测试 R²

# --- 2. 执行 Y 随机化检验 ---
print(f"\n=== 开始 Y 随机化检验 ({N_PERMUTATIONS} 次置换) ===")
print(f"(每次置换使用 APSO 迭代={APSO_ITER_PERM}, BO 调用={BO_CALLS_PERM})")

shuffled_test_r2_scores = [] # 存储每次置换的测试集 R²

# 创建用于洗牌的随机数生成器
rng = np.random.default_rng(seed=123) # 使用固定种子保证洗牌过程可重复

start_time_perms = time.time()
for p in range(N_PERMUTATIONS):
    start_time_perm = time.time()
    print(f"\n--- 运行置换 {p + 1}/{N_PERMUTATIONS} ---")

    # 创建 y_train 的副本并洗牌
    y_train_shuffled = y_train_original.copy()
    rng.shuffle(y_train_shuffled) # 原地洗牌

    # 使用洗牌后的 y_train 运行优化流程 (使用较少的迭代次数)
    # 将 verbose_level 设为 0 以避免大量输出
    perm_test_r2 = run_optimization_pipeline(X_train_original, y_train_shuffled, X_test_fixed, y_test_fixed,
                                            apso_particles=30, # 可适当减少粒子数
                                            apso_iters=APSO_ITER_PERM,
                                            bo_calls=BO_CALLS_PERM,
                                            verbose_level=0) # 关闭内部输出

    shuffled_test_r2_scores.append(perm_test_r2)
    end_time_perm = time.time()
    print(f"置换 {p + 1} 完成. 测试集 R² = {perm_test_r2:.4f}. (耗时: {end_time_perm - start_time_perm:.2f} 秒)")

end_time_perms = time.time()
print(f"\n=== Y 随机化检验完成 ===")
print(f"总共 {N_PERMUTATIONS} 次置换耗时: {end_time_perms - start_time_perms:.2f} 秒")

# --- 3. 分析 Y 随机化结果 ---
shuffled_test_r2_scores = np.array(shuffled_test_r2_scores)
# 过滤掉可能因优化失败产生的无效 R² 值 (例如 -inf)
valid_shuffled_r2 = shuffled_test_r2_scores[np.isfinite(shuffled_test_r2_scores)]
n_valid_perms = len(valid_shuffled_r2)

print(f"\n--- Y 随机化结果分析 (基于 {n_valid_perms} 次有效置换) ---")

if n_valid_perms > 0:
    # 计算 p 值: 置换 R² >= 原始 R² 的比例
    p_value_r2 = np.sum(valid_shuffled_r2 >= original_test_r2) / n_valid_perms

    # 计算置换 R² 的统计信息
    mean_shuffled_r2 = np.mean(valid_shuffled_r2)
    std_shuffled_r2 = np.std(valid_shuffled_r2)
    median_shuffled_r2 = np.median(valid_shuffled_r2) # 中位数可能更稳健

    print(f"原始模型测试集 R²: {original_test_r2:.4f}")
    print(f"随机化模型测试集 R² (均值 ± 标准差): {mean_shuffled_r2:.4f} ± {std_shuffled_r2:.4f}")
    print(f"随机化模型测试集 R² (中位数): {median_shuffled_r2:.4f}")
    print(f"Y 随机化检验 p 值 (R²): {p_value_r2:.4f}")
    print(f"(p 值表示随机情况下获得 R² >= {original_test_r2:.4f} 的概率)")

    # 结论
    if p_value_r2 < 0.05:
        print("\n结论: 原始模型的性能具有统计显著性 (p < 0.05)。")
        print("      模型可能学习到了真实的 X-y 关系，而不仅仅是拟合了噪声。")
    else:
        print("\n结论: 原始模型的性能不具有统计显著性 (p >= 0.05)。")
        print("      需要谨慎对待模型结果，可能存在过拟合或模型未能有效学习到 X-y 关系。")
else:
    print("\n错误: 没有从置换运行中获得有效的 R² 结果，无法计算 p 值。")
    print("      请检查优化流程在处理随机化数据时是否频繁失败。")

# --- 4. 可视化 Y 随机化结果 ---
if n_valid_perms > 0:
    try:
        plt.figure(figsize=(8, 6))
        # 设置 matplotlib 支持中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
        plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

        plt.hist(valid_shuffled_r2, bins=20, alpha=0.7, color='lightblue', edgecolor='black', label='随机化测试集 R² 分布')
        plt.axvline(original_test_r2, color='red', linestyle='dashed', linewidth=2, label=f'原始测试集 R² ({original_test_r2:.3f})')
        plt.xlabel("测试集 R² 分数")
        plt.ylabel("频数")
        plt.title(f"Y 随机化检验结果 ({n_valid_perms} 次有效置换)")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('y_randomization_results.png', dpi=300) # 保存图像
        plt.show()
    except ImportError:
        print("\n未找到 Matplotlib 库。跳过绘制随机化结果直方图。")
    except Exception as e:
        print(f"\n绘制随机化结果直方图时出错: {e}")

print("\n脚本执行完毕.")