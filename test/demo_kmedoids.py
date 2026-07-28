import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn_extra.cluster import KMedoids

# 自动处理路径问题：确保能从根目录导入算法和工具包
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from algorithms.labelingCluster import labelCluster
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


# ------------------------------------------------------------------------------
# 1. 实验运行函数 (对齐 32 列指标结构)
# ------------------------------------------------------------------------------
def run_kmedoids_iteration(X_package, LOC, save_path, project_name, model_name, randseed):
    """
    运行单次 K-Medoids 迭代
    """
    # 1. 数据切分 (Bootstrap)
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)

    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]

    # 2. 预处理 (与主实验对齐：使用 scale)
    test_X_raw = test_data[0]
    test_X_scaled = preprocessing.scale(test_X_raw)

    start_time = time.perf_counter()

    try:
        # 3. 运行 K-Medoids 算法
        # n_clusters=2 (有缺陷 vs 无缺陷)
        kmedoids = KMedoids(n_clusters=2, random_state=randseed).fit(test_X_scaled)
        clus_labels = kmedoids.labels_

        # 使用 labelCluster 确定哪个簇是缺陷簇 (通常基于特征均值)
        predict_y = labelCluster(test_X_scaled, clus_labels)

        # 将标签转换为浮点分数用于 rank_measure
        pred_scores = predict_y.astype(float)

        exec_time = time.perf_counter() - start_time

        # 4. 指标计算 (对齐 32 列)
        y_true = np.array(test_label).astype(int)
        y_pred = np.array(predict_y).astype(int)

        # m1: 基础分类指标 (10个)
        m1 = performanceMeasure.get_measure(y_true, y_pred)

        # m2 & m3: 努力感知指标 (21个)
        res_rank = rankMeasurev2.rank_measure(pred_scores, t_loc, test_label)
        m2 = res_rank[:11]  # c系列 (包含 Popt)
        m3 = res_rank[11:]  # m系列

        # 5. 合并结果 (32列: 10 + 11 + 10 + 1)
        full_measures = list(m1) + list(m2) + list(m3) + [exec_time]

        # 6. 保存结果
        res_dir = create_dir(os.path.join(save_path, model_name + '_results'))
        save_results(os.path.join(res_dir, project_name), full_measures)

    except Exception as e:
        print(f"  Iteration {randseed} Error: {e}")


# ------------------------------------------------------------------------------
# 2. 主循环逻辑
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 【实验配置】
    Reps = 100  # 对齐主实验的重复次数
    data_dir = '../data/'
    save_path_root = '../result_USDP/KMedoids/'
    model_tag = 'INTC_KMedoids'

    project_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for file in project_list:
        p_name = file[:-4]
        raw_df = pd.read_csv(os.path.join(data_dir, file))

        # 1. 识别 LOC 与 标签
        loc_candidates = ['CountLineCode', 'loc', 'LOC']
        loc_col = next((c for c in loc_candidates if c in raw_df.columns), raw_df.columns[0])
        LOC_series = raw_df[loc_col]

        label_col = raw_df.columns[-1]
        y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)

        # 提取全部原始特征 (去掉最后一列标签)
        X_raw = raw_df.iloc[:, :-1]

        # 2. 封装数据包 (跳过特征选择，直接把完整的 X_raw 喂给后续的交叉验证与模型)
        X_package = [X_raw, y_series]

        print(f"\n>>> 开始运行 {model_tag} | 项目: {p_name} | 最终特征数: {X_raw.shape[1]}")

        # 3. 运行 Reps 轮实验
        for r in range(Reps):
            run_kmedoids_iteration(X_package, LOC_series, save_path_root, p_name, model_tag, r)
            if (r + 1) % 10 == 0:
                print(f"  Progress: {r + 1}/{Reps} rounds.")

    print("\n✅ K-Medoids 对比实验运行完成！")