import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing

# 自动处理路径问题：确保能找到 algorithms 和 utilities
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


# ------------------------------------------------------------------------------
# 1. CLA 算法核心逻辑
# ------------------------------------------------------------------------------
def cla_labeling(test_data_scaled):
    """
    CLA 核心逻辑：基于特征中位数二值化后进行聚类打标
    """
    X = test_data_scaled
    n, dim = X.shape

    # 1. 计算每个特征的中位数
    thresholds = np.median(X, axis=0)

    # 2. 特征二值化：大于中位数为1，否则为0
    binary_matrix = (X > thresholds).astype(int)

    # 3. 统计每个样本“高风险”特征的数量 (Score)
    # 在 CLA 中，满足阈值的特征越多，风险越高
    scores = np.sum(binary_matrix, axis=1).astype(float)

    # 4. 聚类划分逻辑
    unique_counts = np.unique(scores)
    num_clusters = len(unique_counts)

    # 将样本按 score 分组（模拟聚类）
    clusters = [np.where(scores == u)[0] for u in unique_counts]

    # 划分阈值 k：后一半的聚类划分为缺陷 (1)
    k = int(np.ceil(num_clusters / 2))

    predict_y = np.zeros(n)
    # 将高分聚类索引合并
    defective_indices = np.concatenate(clusters[k:]) if k < num_clusters else np.array([])
    if len(defective_indices) > 0:
        predict_y[defective_indices] = 1

    return predict_y, scores


# ------------------------------------------------------------------------------
# 2. 实验运行函数 (与主实验流程对齐)
# ------------------------------------------------------------------------------
def run_cla_iteration(X_data, LOC, save_path, project_name, model_name, randseed):
    """
    执行单次 CLA 实验循环
    """
    # 1. Bootstrap 采样 (与主实验 seed 一致)
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_data, randseed)

    # 2. 提取测试集 LOC
    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]

    # 3. 数据预处理：【关键】与主实验对齐使用 scale
    # test_data[0] 为特征 DataFrame
    test_X_scaled = preprocessing.scale(test_data[0])

    start_time = time.perf_counter()

    # 4. 运行算法
    pred_y, pred_scores = cla_labeling(test_X_scaled)

    exec_time = time.perf_counter() - start_time

    # 5. 计算指标 (32列)
    y_true = np.array(test_label).astype(int)
    y_pred = np.array(pred_y).astype(int)

    # m1: 分类指标 (10个)
    m1 = performanceMeasure.get_measure(y_true, y_pred)

    # m2 & m3: 努力感知指标 (21个: 11个c系列 + 10个m系列)
    # rank_measure 内部会处理 20%LOC 和 20%Modules 的切分
    res_rank = rankMeasurev2.rank_measure(pred_scores, t_loc, test_label)
    m2 = res_rank[:11]
    m3 = res_rank[11:]

    # 6. 整合并保存
    full_measures = list(m1) + list(m2) + list(m3) + [exec_time]

    res_dir = create_dir(os.path.join(save_path, model_name + '_results'))
    save_results(os.path.join(res_dir, project_name), full_measures)


# ------------------------------------------------------------------------------
# 3. 主程序
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 配置信息
    Reps = 100
    # 根据你服务器运行位置调整相对路径
    data_dir = '../data/'
    save_path_root = '../result_USDP/CLA/'
    model_tag = 'INTC_CLA'

    project_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for file in project_list:
        p_name = file[:-4]
        raw_df = pd.read_csv(os.path.join(data_dir, file))

        # 1. 准备标签与 LOC
        # 寻找 LOC 列名
        loc_candidates = ['CountLineCode', 'loc', 'LOC']
        loc_col = next((c for c in loc_candidates if c in raw_df.columns), raw_df.columns[0])
        LOC_series = raw_df[loc_col]

        # 标签二值化 (缺陷 > 0 为 1)
        label_col = raw_df.columns[-1]
        y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)

        # 提取全部原始特征 (去掉最后一列标签)
        X_raw = raw_df.iloc[:, :-1]

        # 2. 封装数据 (不进行特征选择，直接使用原始特征 X_raw)
        X_package = [X_raw, y_series]

        print(f"\n开始运行 {model_tag} 实验 [Project: {p_name}], 特征数: {X_raw.shape[1]}")

        # 3. 运行重复实验
        for r in range(Reps):
            run_cla_iteration(X_package, LOC_series, save_path_root, p_name, model_tag, r)
            if (r + 1) % 10 == 0:
                print(f"  Progress: {r + 1}/{Reps} rounds completed.")

    print("\nCLA 对比实验运行完毕，结果已保存。")