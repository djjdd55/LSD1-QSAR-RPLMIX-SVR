import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler # 用于归一化

# --- 配置参数 ---
TRAINING_DATA_FILE = 'Xtrain.csv'
NEW_COMPOUNDS_FILE = 'x_new.csv'
MIN_MAX_RANGE = (0, 1) # 归一化到的范围，通常是[0,1]

# --- 1. 读取原始数据 ---
try:
    X_train_orig_df = pd.read_csv(TRAINING_DATA_FILE, header=None)
    X_train_orig = X_train_orig_df.values
    print(f"成功读取原始训练数据文件: {TRAINING_DATA_FILE}")
except FileNotFoundError:
    print(f"错误: 训练数据文件 '{TRAINING_DATA_FILE}' 未找到。")
    exit()
# ... (可以加入更多文件读取错误处理) ...

try:
    X_new_orig_df = pd.read_csv(NEW_COMPOUNDS_FILE, header=None)
    X_new_orig = X_new_orig_df.values
    print(f"成功读取原始新化合物数据文件: {NEW_COMPOUNDS_FILE}")
except FileNotFoundError:
    print(f"错误: 新化合物数据文件 '{NEW_COMPOUNDS_FILE}' 未找到。")
    exit()
# ... (可以加入更多文件读取错误处理) ...

# --- 2. 数据归一化 ---
# 初始化MinMaxScaler
scaler_minmax = MinMaxScaler(feature_range=MIN_MAX_RANGE)

# 使用训练数据拟合scaler并转换训练数据
# scaler_minmax.fit() 会计算训练数据的最小值和最大值
# scaler_minmax.transform() 会用计算出的min/max进行归一化
try:
    X_train_scaled = scaler_minmax.fit_transform(X_train_orig)
    print("\n训练数据已归一化。")
    print(f"归一化后训练集最小值 (近似{MIN_MAX_RANGE[0]}): {np.min(X_train_scaled, axis=0)}")
    print(f"归一化后训练集最大值 (近似{MIN_MAX_RANGE[1]}): {np.max(X_train_scaled, axis=0)}")
except ValueError as e:
    print(f"错误: 归一化训练数据时发生错误: {e}")
    print("这可能发生在数据列全为常数的情况下 (max-min会是0)。")
    exit()


# 使用从训练数据学习到的scaler转换新化合物数据
try:
    X_new_scaled = scaler_minmax.transform(X_new_orig)
    print("新化合物数据已使用训练集的参数进行归一化。")
except ValueError as e:
    print(f"错误: 归一化新化合物数据时发生错误: {e}")
    exit()


# --- 3. 参数计算 (基于归一化后的数据) ---
n = X_train_scaled.shape[0]
p = X_train_scaled.shape[1]

if X_new_scaled.shape[1] != p:
    print(f"错误: 归一化后的新化合物描述符数量与训练集不匹配。")
    exit()
if n <= 0:
    print(f"错误: 训练集化合物数量 (n={n}) 必须为正数。")
    exit()

h_star = (3 * p) / n

print(f"\n--- 参数信息 (基于归一化数据) ---")
print(f"训练集化合物数 (n): {n}")
print(f"描述符数 (p): {p}")
print(f"警告杠杆值 (h* = 3*p/n): {h_star:.4f}")

# --- 4. 计算 (X_scaled^T * X_scaled)^-1 ---
XTX_scaled = np.dot(X_train_scaled.T, X_train_scaled)
try:
    XTX_inv_scaled = np.linalg.inv(XTX_scaled)
    print("成功计算 (X_scaled^T * X_scaled)^-1.")
except np.linalg.LinAlgError:
    print("\n错误: 矩阵 (X_scaled^T * X_scaled) 是奇异矩阵，不可逆。")
    print("即使在归一化后，如果描述符间仍存在完美共线性，也可能发生此问题。")
    exit()

# --- 5. 计算新化合物的杠杆值 (基于归一化数据) 并进行判断 ---
leverages_new_compounds_scaled = []
print(f"\n--- 新化合物适用域分析 (基于归一化数据, 阈值 h* = {h_star:.4f}) ---")

if X_new_scaled.shape[0] == 0:
    print("没有新化合物需要分析。")
else:
    for i in range(X_new_scaled.shape[0]):
        x_i_new_scaled = X_new_scaled[i, :]
        leverage_i_scaled = np.dot(np.dot(x_i_new_scaled, XTX_inv_scaled), x_i_new_scaled.T)
        leverages_new_compounds_scaled.append(leverage_i_scaled)

        status = "内部 (INSIDE)" if leverage_i_scaled <= h_star else "外部 (OUTSIDE)"
        comparison = "<=" if leverage_i_scaled <= h_star else ">"
        print(f"新化合物 {i+1}: 杠杆值 = {leverage_i_scaled:.4f} ({comparison} {h_star:.4f}) -> 适用域 {status}")

print("\n归一化数据AD分析完成。")