import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing

# 自动处理路径问题：确保能从根目录导入算法和工具包
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from algorithms.TCL import TCL
from algorithms.TCLP import TCLP
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


# ------------------------------------------------------------------------------
# 1. 实验运行函数 (对齐 32 列指标)
# ------------------------------------------------------------------------------
def run_tclp_iteration(X_package, LOC, save_path, project_name, model_name, randseed):
    """
    运行单次 TCLP 迭代 (包含 TCL 初始化与 TCLP 修剪)
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

    # 转回 DataFrame 以保持 TCLP 可能需要的列结构
    test_X_df = pd.DataFrame(test_X_scaled, columns=test_X_raw.columns)

    start_time = time.perf_counter()

    try:
        # 3. 运行算法组合：TCL 获取初始标签 -> TCLP 进行修剪优化
        # TCL 通常返回聚类标签和计算出的阈值
        clus_label, metric_threshold = TCL(test_X_df)

        # TCLP 根据阈值和初始标签进行过滤/修剪
        predict_y = TCLP(test_X_df, clus_label, metric_threshold)

        # 将标签转换为浮点分数用于 rank_measure 计算
        pred_scores = predict_y.astype(float)

        exec_time = time.perf_counter() - start_time

        # 4. 指标计算 (对齐 32 列)
        y_true = np.array(test_label).astype(int)
        y_pred = np.array(predict_y).astype(int)

        # m1: 基础分类指标 (10个)
        m1 = performanceMeasure.get_measure(y_true, y_pred)

        # m2 & m3: 努力感知指标 (21个)
        res_rank = rankMeasurev2.rank_measure(pred_scores, t_loc, test_label)
        m2 = res_rank[:11]  # c系列
        m3 = res_rank[11:]  # m系列

        # 5. 合并结果 (32列) Full Measures
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
    Reps = 100
    data_dir = '../data/'
    save_path_root = '../result_USDP/TCLP/'
    model_tag = 'INTC_TCLP'

    project_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for file in project_list:
        p_name = file[:-4]
        raw_df = pd.read_csv(os.path.join(data_dir, file))

        # 1. 识别 LOC 与 标签
        # 优先寻找 CountLineCode，否则找 loc
        loc_candidates = ['CountLineCode', 'loc', 'LOC']
        loc_col = next((c for c in loc_candidates if c in raw_df.columns), raw_df.columns[0])
        LOC_series = raw_df[loc_col]

        label_col = raw_df.columns[-1]
        y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)
        X_raw = raw_df.iloc[:, :-1]

        # 2. 准备数据包 (跳过特征选择，直接透传完整的原始特征 X_raw)
        X_package = [X_raw, y_series]

        print(f"\n>>> 开始运行 {model_tag} | 项目: {p_name} | 原始特征数: {X_raw.shape[1]}")

        # 3. 运行 Reps 轮实验
        for r in range(Reps):
            run_tclp_iteration(X_package, LOC_series, save_path_root, p_name, model_tag, r)
            if (r + 1) % 10 == 0:
                print(f"  Status: {r + 1}/{Reps} rounds.")

    print("\n✅ TCLP 对比实验全部完成！")