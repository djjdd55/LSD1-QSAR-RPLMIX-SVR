import pandas as pd
import numpy as np
import re
import time
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import catboost as cb
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from pyswarm import pso

# --- 全局变量 ---
CSV_FILE_PATH = 'descriptors data.csv'
TARGET_COLUMN = 'TEZHENG'
NUM_TOP_FEATURES = 8
CORRELATION_THRESHOLD = 0.8
VARIANCE_THRESHOLD = 0.01
RANDOM_STATE = 14

PSO_N_PARTICLES = 50
PSO_MAX_ITER = 20


# preprocess_descriptors 函数 
def preprocess_descriptors(csv_file, target_column='TEZHENG', variance_threshold=0.01, correlation_threshold=0.99,
                           prioritized_list=None):
    if prioritized_list is None:
        prioritized_list = PRIORITIZED_FEATURES_FROM_IMAGE
    print(f"--- 开始预处理 ---")
    print(f"正在读取文件: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"错误: 文件 {csv_file} 未找到。请检查路径。")
        return None, None, [], []
    print(f"成功读取数据，共 {df.shape[0]} 行，{df.shape[1]} 列")

    if target_column not in df.columns:
        print(f"错误: 目标列 '{target_column}' 在CSV文件中未找到。可用列: {df.columns.tolist()}")
        return None, None, [], []
    if df.shape[0] < 2:
        print(f"错误: 数据行数不足 ({df.shape[0]}行)，无法计算方差或相关性。至少需要2行数据。")
        return None, None, [], []

    X = df.drop(columns=[target_column])
    y_target_series = df[target_column]
    initial_features_count = X.shape[1]
    print(f"初始特征数量: {initial_features_count}")

    non_convertible_cols = []
    for col in list(X.columns):
        try:
            X[col] = pd.to_numeric(X[col], errors='raise')
        except (ValueError, TypeError):
            non_convertible_cols.append(col)
            X = X.drop(columns=[col])
    if non_convertible_cols:
        print(f"警告: 以下列包含无法转换为数值类型的值，将从特征集中移除: {non_convertible_cols}")
        print(f"移除非数值可转换列后，剩余特征数量: {X.shape[1]}")

    if X.empty:
        print("错误: 移除无法转换为数值的列后，没有剩余特征。")
        return None, y_target_series, non_convertible_cols, []

    if X.isnull().any().any():
        print("警告: 特征数据中存在NaN值，将使用列均值填充。")
        for col_name in X.columns[X.isnull().any()]:
            X.loc[:, col_name] = X[col_name].fillna(X[col_name].mean())

    print("\n步骤1: 去除变化很小的描述符...")
    X = X.select_dtypes(include=np.number)
    if X.empty:
        print("错误: 在转换为数值类型后，没有特征可用于方差计算。")
        return None, y_target_series, non_convertible_cols, []

    variances = X.var()
    low_var_features = variances[variances < variance_threshold].index.tolist()
    X_processed = X.drop(columns=low_var_features, errors='ignore')
    removed_low_var = low_var_features
    print(
        f"因方差低 (<{variance_threshold}) 移除了 {len(removed_low_var)} 个描述符: {removed_low_var if removed_low_var else '无'}")
    print(f"移除低方差特征后剩余特征数量: {X_processed.shape[1]}")

    if X_processed.empty:
        print("移除低方差特征后没有剩余特征。")
        return X_processed, y_target_series, removed_low_var, []

    print(f"\n步骤2: 去除高度相关的描述符(相关性阈值>{correlation_threshold})...")
    X_processed = X_processed.select_dtypes(include=np.number)

    if not pd.api.types.is_numeric_dtype(y_target_series):
        try:
            y_target_series = pd.to_numeric(y_target_series, errors='raise')
        except (ValueError, TypeError):
            print(f"错误: 目标列 '{target_column}' 无法转换为数值类型。")
            return X_processed, y_target_series, removed_low_var, []
    if y_target_series.isnull().any():
        y_target_series = y_target_series.fillna(y_target_series.mean())

    target_correlation = {}
    for column in X_processed.columns:
        if X_processed[column].var() == 0 or y_target_series.var() == 0:
            target_correlation[column] = 0
        else:
            feature_data_no_nan = X_processed[column].fillna(X_processed[column].mean())
            target_data_no_nan = y_target_series
            if len(feature_data_no_nan) != len(target_data_no_nan):
                print(f"警告: 特征 '{column}' 和目标长度不匹配，跳过相关性计算。")
                target_correlation[column] = 0
                continue
            try:
                corr_val = np.corrcoef(feature_data_no_nan, target_data_no_nan)[0, 1]
                target_correlation[column] = abs(corr_val if not np.isnan(corr_val) else 0)
            except Exception as e:
                print(f"计算特征 '{column}' 与目标的相关性时出错: {e}. 设为0。")
                target_correlation[column] = 0

    feature_info_for_sorting = []
    for col in X_processed.columns:
        is_prioritized = 1 if col in prioritized_list else 0
        feature_info_for_sorting.append((col, target_correlation.get(col, 0), is_prioritized))

    feature_info_for_sorting.sort(key=lambda x: (x[2], x[1]), reverse=True)
    sorted_features_overall_priority = [info[0] for info in feature_info_for_sorting]

    correlation_matrix_processed = X_processed.corr().abs()
    to_drop_collinear = set()
    removed_collinear_info = []

    for i in range(len(sorted_features_overall_priority)):
        col_i = sorted_features_overall_priority[i]
        if col_i in to_drop_collinear: continue
        col_i_is_prioritized_img = col_i in prioritized_list
        col_i_target_corr = target_correlation.get(col_i, 0)
        for j in range(i + 1, len(sorted_features_overall_priority)):
            col_j = sorted_features_overall_priority[j]
            if col_j in to_drop_collinear: continue
            if col_i in correlation_matrix_processed.index and col_j in correlation_matrix_processed.columns:
                corr_val_ij = correlation_matrix_processed.loc[col_i, col_j]
            else:
                corr_val_ij = 0

            if corr_val_ij > correlation_threshold:
                col_j_is_prioritized_img = col_j in prioritized_list
                col_j_target_corr = target_correlation.get(col_j, 0)
                feature_to_drop, feature_to_keep, reason_for_choice = (None, None, "")
                if col_i_is_prioritized_img and not col_j_is_prioritized_img:
                    feature_to_drop, feature_to_keep = col_j, col_i
                    reason_for_choice = f"保持 '{col_i}' (图片优先), 丢弃 '{col_j}'"
                elif not col_i_is_prioritized_img and col_j_is_prioritized_img:
                    feature_to_drop, feature_to_keep = col_i, col_j
                    reason_for_choice = f"保持 '{col_j}' (图片优先), 丢弃 '{col_i}'"
                elif (col_i_is_prioritized_img and col_j_is_prioritized_img) or \
                        (not col_i_is_prioritized_img and not col_j_is_prioritized_img):
                    if col_i_target_corr >= col_j_target_corr:
                        feature_to_drop, feature_to_keep = col_j, col_i
                        prefix = "两者均图片优先, " if col_i_is_prioritized_img else "均非图片优先, "
                        reason_for_choice = prefix + f"保持 '{col_i}' (与目标相关性更高或同等), 丢弃 '{col_j}'"
                    else:
                        feature_to_drop, feature_to_keep = col_i, col_j
                        prefix = "两者均图片优先, " if col_i_is_prioritized_img else "均非图片优先, "
                        reason_for_choice = prefix + f"保持 '{col_j}' (与目标相关性更高), 丢弃 '{col_i}'"

                to_drop_collinear.add(feature_to_drop)
                removed_collinear_info.append({
                    'dropped': feature_to_drop, 'kept': feature_to_keep, 'reason': reason_for_choice,
                    'correlation_between_features': corr_val_ij,
                    'corr_target_dropped': target_correlation.get(feature_to_drop, 0),
                    'corr_target_kept': target_correlation.get(feature_to_keep, 0)})

    final_features_to_keep = [col for col in X_processed.columns if col not in to_drop_collinear]
    X_processed = X_processed[final_features_to_keep]

    print(f"因共线性 (阈值>{correlation_threshold}) 并根据自定义优先级移除了 {len(to_drop_collinear)} 个描述符。")
    if removed_collinear_info and len(removed_collinear_info) > 0:
        print("详细移除信息 (丢弃的特征, 保留的特征, 原因, 两者相关性, ...) - 最多显示5条:")
        for item in removed_collinear_info[:5]:
            print(f"  - 丢弃: {item['dropped']} (与目标相关性: {item['corr_target_dropped']:.3f}), "
                  f"保留: {item['kept']} (与目标相关性: {item['corr_target_kept']:.3f}), "
                  f"两者相关性: {item['correlation_between_features']:.3f}, 原因: {item['reason']}")
        if len(removed_collinear_info) > 5:
            print(f"  ... 等等 (共 {len(removed_collinear_info)} 条)")
    else:
        print("没有因共线性移除特征。")

    print(f"预处理后最终特征数量: {X_processed.shape[1]}")
    print(f"--- 预处理结束 ---")
    return X_processed, y_target_series, removed_low_var, removed_collinear_info


def sanitize_feature_names(df_with_features):
    original_cols = df_with_features.columns.tolist()
    name_mapping_s_to_o = {}
    name_mapping_o_to_s = {}
    sanitized_cols = []
    for i, col in enumerate(original_cols):
        sanitized_col = re.sub(r"[\[\]<]", "_", str(col))
        sanitized_col = re.sub(r'[^A-Za-z0-9_]+', '_', sanitized_col)
        original_sanitized_col = sanitized_col
        count = 1
        while sanitized_col in sanitized_cols:
            sanitized_col = f"{original_sanitized_col}_{count}"
            count += 1
        sanitized_cols.append(sanitized_col)
        name_mapping_s_to_o[sanitized_col] = col
        name_mapping_o_to_s[col] = sanitized_col
    df_copy = df_with_features.copy()
    df_copy.columns = sanitized_cols
    return df_copy, name_mapping_s_to_o, name_mapping_o_to_s


# --- PSO调优相关函数 (与之前相同) ---
_PSO_GLOBAL_X = None
_PSO_GLOBAL_Y = None
_PSO_MODEL_TYPE = None
_PSO_PARAM_NAMES = None
_PSO_RANDOM_STATE = None


def pso_objective_function(params):
    global _PSO_GLOBAL_X, _PSO_GLOBAL_Y, _PSO_MODEL_TYPE, _PSO_PARAM_NAMES, _PSO_RANDOM_STATE
    param_dict = dict(zip(_PSO_PARAM_NAMES, params))
    # Integer param conversion
    if 'n_estimators' in param_dict: param_dict['n_estimators'] = int(param_dict['n_estimators'])
    if 'iterations' in param_dict: param_dict['iterations'] = int(param_dict['iterations'])
    if 'max_depth' in param_dict: param_dict['max_depth'] = int(param_dict['max_depth'])
    if 'depth' in param_dict: param_dict['depth'] = int(param_dict['depth'])
    if 'num_leaves' in param_dict: param_dict['num_leaves'] = int(param_dict['num_leaves'])
    if 'min_samples_split' in param_dict: param_dict['min_samples_split'] = int(param_dict['min_samples_split'])
    if 'min_samples_leaf' in param_dict: param_dict['min_samples_leaf'] = int(param_dict['min_samples_leaf'])
    if 'border_count' in param_dict: param_dict['border_count'] = int(param_dict['border_count'])
    model = None
    if _PSO_MODEL_TYPE == 'RandomForest':
        model = RandomForestRegressor(**param_dict, random_state=_PSO_RANDOM_STATE, n_jobs=-1)
    # XGBoost 块已移除
    elif _PSO_MODEL_TYPE == 'LightGBM':
        # *** 此处是修复点 ***
        if 'subsample' in param_dict and param_dict['subsample'] > 1.0: param_dict['subsample'] = 1.0
        if 'colsample_bytree' in param_dict and param_dict['colsample_bytree'] > 1.0: param_dict['colsample_bytree'] = 1.0
        model = lgb.LGBMRegressor(**param_dict, random_state=_PSO_RANDOM_STATE, n_jobs=-1, verbosity=-1)
    elif _PSO_MODEL_TYPE == 'CatBoost':
        model = cb.CatBoostRegressor(**param_dict, random_state=_PSO_RANDOM_STATE, verbose=0, loss_function='RMSE')
    else:
        raise ValueError("Unsupported model type for PSO")
    try:
        model.fit(_PSO_GLOBAL_X, _PSO_GLOBAL_Y)
        predictions = model.predict(_PSO_GLOBAL_X)
        mse = mean_squared_error(_PSO_GLOBAL_Y, predictions)
        return mse
    except Exception as e:
        return np.inf


def optimize_params_with_pso(model_type, X_train, y_train, param_config, random_state):
    global _PSO_GLOBAL_X, _PSO_GLOBAL_Y, _PSO_MODEL_TYPE, _PSO_PARAM_NAMES, _PSO_RANDOM_STATE
    _PSO_GLOBAL_X, _PSO_GLOBAL_Y, _PSO_MODEL_TYPE, _PSO_RANDOM_STATE = X_train, y_train, model_type, random_state
    _PSO_PARAM_NAMES = list(param_config.keys())
    lb = [param_config[name][0] for name in _PSO_PARAM_NAMES]
    ub = [param_config[name][1] for name in _PSO_PARAM_NAMES]
    print(f"开始为 {model_type} 进行PSO参数调优 (粒子数: {PSO_N_PARTICLES}, 最大迭代: {PSO_MAX_ITER}, 无CV)...")
    start_time = time.time()
    best_params_values, min_mse = pso(pso_objective_function, lb, ub, swarmsize=PSO_N_PARTICLES, maxiter=PSO_MAX_ITER,
                                      debug=False)
    end_time = time.time()
    print(f"{model_type} PSO调优完成，耗时: {end_time - start_time:.2f} 秒. 最佳训练集MSE: {min_mse:.4f}")
    best_params_dict = dict(zip(_PSO_PARAM_NAMES, best_params_values))
    # Integer param conversion (same as in objective func)
    if 'n_estimators' in best_params_dict: best_params_dict['n_estimators'] = int(best_params_dict['n_estimators'])
    if 'iterations' in best_params_dict: best_params_dict['iterations'] = int(best_params_dict['iterations'])
    if 'max_depth' in best_params_dict: best_params_dict['max_depth'] = int(best_params_dict['max_depth'])
    if 'depth' in best_params_dict: best_params_dict['depth'] = int(best_params_dict['depth'])
    if 'num_leaves' in best_params_dict: best_params_dict['num_leaves'] = int(best_params_dict['num_leaves'])
    if 'min_samples_split' in best_params_dict: best_params_dict['min_samples_split'] = int(
        best_params_dict['min_samples_split'])
    if 'min_samples_leaf' in best_params_dict: best_params_dict['min_samples_leaf'] = int(
        best_params_dict['min_samples_leaf'])
    if 'border_count' in best_params_dict: best_params_dict['border_count'] = int(best_params_dict['border_count'])
    print(f"{model_type} 最佳参数: {best_params_dict}")
    return best_params_dict


# --- 特征选择主函数 (集成投票 + RFE，RFE使用LightGBM) ---
def select_features_ensemble_and_rfe_voting(X_processed, y, num_top_features, random_state):
    print(f"\n--- 开始集成投票+RFE特征选择 (选择前 {num_top_features} 个, 模型参数PSO调优 - 无CV) ---")

    X_numeric = X_processed.select_dtypes(include=np.number).copy()
    if X_numeric.isnull().any().any():
        for col in X_numeric.columns[X_numeric.isnull().any()]:
            X_numeric.loc[:, col] = X_numeric[col].fillna(X_numeric[col].mean())
    if X_numeric.empty: print("错误: 清理后没有数值特征。"); return [], pd.DataFrame()

    y_filled = y.copy()
    if not pd.api.types.is_numeric_dtype(y_filled):
        try:
            y_filled = pd.to_numeric(y_filled, errors='raise')
        except (ValueError, TypeError):
            print(f"错误: 目标列无法转为数值。"); return [], pd.DataFrame()
    if y_filled.isnull().any(): y_filled = y_filled.fillna(y_filled.mean())

    X_sanitized, name_map_s_to_o, name_map_o_to_s = sanitize_feature_names(X_numeric.copy())
    original_feature_names_ordered = X_numeric.columns.tolist()

    param_spaces = {
        "RandomForest": {'n_estimators': [50, 200], 'max_depth': [3, 15], 'min_samples_split': [2, 15],
                         'min_samples_leaf': [1, 10]},
        # "XGBoost" 参数空间已移除
        "LightGBM": {'n_estimators': [50, 200], 'learning_rate': [0.005, 0.3], 'num_leaves': [10, 60],
                     'subsample': [0.5, 1.0], 'colsample_bytree': [0.5, 1.0]},
        "CatBoost": {'iterations': [50, 200], 'learning_rate': [0.005, 0.3], 'depth': [2, 10], 'l2_leaf_reg': [1, 15]}
    }

    tuned_model_params = {}
    print("\n--- 模型超参数调优 (PSO 无CV) ---")
    for model_name, p_space in param_spaces.items():
        current_X_for_pso = X_numeric if model_name == "RandomForest" else X_sanitized
        tuned_model_params[model_name] = optimize_params_with_pso(
            model_name, current_X_for_pso, y_filled, p_space, random_state
        )

    # --- 1. 三个模型的重要性投票 ---
    models_for_importance = {
        "RandomForest": RandomForestRegressor(**tuned_model_params["RandomForest"], random_state=random_state,
                                              n_jobs=-1),
        "LightGBM": lgb.LGBMRegressor(**tuned_model_params["LightGBM"], random_state=random_state, n_jobs=-1,
                                      verbosity=-1),
        "CatBoost": cb.CatBoostRegressor(**tuned_model_params["CatBoost"], random_state=random_state, verbose=0,
                                         loss_function='RMSE')
    }
    all_feature_rankings = pd.DataFrame(index=original_feature_names_ordered)

    for model_name, model_instance in models_for_importance.items():
        print(f"\n使用优化参数训练 {model_name} 以获取特征重要性...")
        current_X_fit = X_numeric if model_name == "RandomForest" else X_sanitized
        try:
            model_instance.fit(current_X_fit, y_filled)
            importances = model_instance.feature_importances_
            current_feature_names_in_model = current_X_fit.columns
            s_importances = pd.Series(importances, index=current_feature_names_in_model)
            importances_original_names = pd.Series(index=original_feature_names_ordered, dtype=float)

            if model_name != "RandomForest":  # Mapped names
                for san_name, imp_val in s_importances.items():
                    orig_name = name_map_s_to_o.get(san_name)
                    if orig_name in importances_original_names.index: importances_original_names[orig_name] = imp_val
            else:  # Original names
                for orig_name, imp_val in s_importances.items():
                    if orig_name in importances_original_names.index: importances_original_names[orig_name] = imp_val

            importances_original_names = importances_original_names.fillna(0)
            temp_df = pd.DataFrame(
                {'feature': importances_original_names.index, 'importance': importances_original_names.values})
            temp_df = temp_df.sort_values('importance', ascending=False).reset_index(drop=True)
            temp_df[f'{model_name}_rank'] = temp_df.index + 1
            for _, row in temp_df.iterrows():
                all_feature_rankings.loc[row['feature'], f'{model_name}_rank'] = row[f'{model_name}_rank']
            print(f"{model_name} 特征重要性 (前5):")
            print(temp_df[['feature', 'importance']].head())
        except Exception as e:
            print(f"{model_name} 模型训练或重要性提取失败: {e}")
            all_feature_rankings[f'{model_name}_rank'] = len(original_feature_names_ordered) + 1

    # --- 2. RFE + Tuned LightGBM 作为第四个投票者 ---
    print("\n--- 执行 RFE + (已调优)LightGBM ---")
    tuned_lgbm_params = tuned_model_params.get("LightGBM")
    if tuned_lgbm_params:
        # 使用调优后的LightGBM参数作为RFE的估计器
        estimator_rfe = lgb.LGBMRegressor(**tuned_lgbm_params, random_state=random_state, n_jobs=-1, verbosity=-1)

        # 确保RFE的输入特征数量不少于num_top_features
        if X_sanitized.shape[1] > num_top_features:
            selector_rfe = RFE(estimator_rfe, n_features_to_select=X_sanitized.shape[1],
                               step=1)  # Select all features initially to get full ranking
            try:
                print("RFE (获取完整排名) 正在拟合...")
                selector_rfe.fit(X_sanitized, y_filled)  # Fit on sanitized names
                rfe_rankings = selector_rfe.ranking_  # Lower is better (1 is most important)

                # 将RFE排名添加到all_feature_rankings
                for i, san_col_name in enumerate(X_sanitized.columns):
                    original_col_name = name_map_s_to_o[san_col_name]
                    all_feature_rankings.loc[original_col_name, 'RFE_LGBM_rank'] = rfe_rankings[i]
                print("RFE 排名已获取。")
            except Exception as e:
                print(f"RFE + Tuned LightGBM 失败: {e}")
                all_feature_rankings['RFE_LGBM_rank'] = len(original_feature_names_ordered) + 1
        elif X_sanitized.shape[1] > 0:
            print(f"特征数 ({X_sanitized.shape[1]}) 不足进行有意义的RFE排名，将对RFE_LGBM_rank赋默认高排名。")
            estimator_rfe.fit(X_sanitized, y_filled)
            rfe_importances = pd.Series(estimator_rfe.feature_importances_, index=X_sanitized.columns)
            rfe_importances_original_names = pd.Series(index=original_feature_names_ordered, dtype=float)
            for san_name, imp_val in rfe_importances.items():
                orig_name = name_map_s_to_o.get(san_name)
                if orig_name in rfe_importances_original_names.index: rfe_importances_original_names[
                    orig_name] = imp_val
            rfe_importances_original_names = rfe_importances_original_names.fillna(0)
            temp_rfe_df = pd.DataFrame(
                {'feature': rfe_importances_original_names.index, 'importance': rfe_importances_original_names.values})
            temp_rfe_df = temp_rfe_df.sort_values('importance', ascending=False).reset_index(drop=True)
            temp_rfe_df['RFE_LGBM_rank'] = temp_rfe_df.index + 1
            for _, row in temp_rfe_df.iterrows():
                all_feature_rankings.loc[row['feature'], 'RFE_LGBM_rank'] = row['RFE_LGBM_rank']
        else:
            all_feature_rankings['RFE_LGBM_rank'] = len(original_feature_names_ordered) + 1

    else:
        print("未能获取LightGBM的调优参数，跳过RFE投票部分。")
        all_feature_rankings['RFE_LGBM_rank'] = len(original_feature_names_ordered) + 1

    # --- 3. 最终投票 ---
    all_feature_rankings = all_feature_rankings.fillna(len(original_feature_names_ordered) + 1)
    all_feature_rankings['avg_rank'] = all_feature_rankings.mean(axis=1)
    final_ranked_features_df = all_feature_rankings.sort_values('avg_rank', ascending=True)

    print("\n--- 所有方法（包括RFE）的特征排名和平均排名 ---")
    print(final_ranked_features_df.head(min(15, len(final_ranked_features_df))).to_string())

    top_features_voted_final = final_ranked_features_df.index[:num_top_features].tolist()

    print(f"\n最终集成投票 (含RFE) 选择出的前 {len(top_features_voted_final)} 个特征已确定。")
    print(f"--- 集成投票+RFE特征选择结束 ---")
    return top_features_voted_final, final_ranked_features_df


if __name__ == '__main__':
    print(f"--- 开始整体特征选择流程 (集成投票含RFE + PSO调优 无CV) ---")
    X_preprocessed, y_target, removed_lv, removed_coll = preprocess_descriptors(
        csv_file=CSV_FILE_PATH, target_column=TARGET_COLUMN,
        variance_threshold=VARIANCE_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        prioritized_list=PRIORITIZED_FEATURES_FROM_IMAGE
    )

    if X_preprocessed is None or X_preprocessed.empty:
        print("\n预处理后没有剩余特征或发生错误。程序终止。")
    elif y_target is None:
        print("\n目标变量未能成功加载或处理。程序终止。")
    else:
        num_remaining_features = X_preprocessed.shape[1]
        print(f"\n预处理后剩余特征列名 (共 {num_remaining_features} 个): ", end="")
        if num_remaining_features <= 20:
            print(X_preprocessed.columns.tolist())
        else:
            print(f"{X_preprocessed.columns.tolist()[:10]} ... {X_preprocessed.columns.tolist()[-10:]}")

        if num_remaining_features == 0:
            print("没有特征可供选择。")
            top_final_features = []
        elif num_remaining_features < NUM_TOP_FEATURES:
            print(
                f"警告: 预处理后特征数 ({num_remaining_features}) 少于要选择的特征数 ({NUM_TOP_FEATURES}). 将选择所有可用特征。")
            top_final_features = X_preprocessed.columns.tolist()
            print(f"\n最终选择的特征（由于数量不足）:")
            for i, feature in enumerate(top_final_features):
                print(f"{i + 1}. {feature}")
        else:
            top_final_features, all_final_rankings = select_features_ensemble_and_rfe_voting(
                X_preprocessed, y_target, NUM_TOP_FEATURES, RANDOM_STATE
            )

            if top_final_features:
                print(
                    f"\n最终由集成投票 (含RFE, 模型PSO调优 - 无CV 后) 选出的最重要的 {len(top_final_features)} 个特征是 (按平均排名):")
                for i, feature in enumerate(top_final_features):
                    if feature in all_final_rankings.index:
                        avg_rank = all_final_rankings.loc[feature, 'avg_rank']
                        print(f"{i + 1}. {feature} (平均排名: {avg_rank:.2f})")
                    else:
                        print(f"{i + 1}. {feature} (排名信息缺失)")
            else:
                print("\n最终集成投票 (含RFE, PSO调优后 - 无CV) 未能选出任何特征。")


    print(f"\n--- 整体特征选择流程结束 ---")
