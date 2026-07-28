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

from algorithms.TCL import TCL
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


# ------------------------------------------------------------------------------
# 1. 实验运行函数 (与主实验及其他对比实验完全对齐)
# ------------------------------------------------------------------------------
def run_tcl_iteration(X_package, LOC, save_path, project_name, model_name, randseed):
    """
    运行单次 TCL 迭代并保存 32 列指标
    """
    # 1. 数据切分 (Bootstrap)
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)

    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]

    # 2. 预处理 (与主实验对齐：使用 scale)
    # test_data[0] 是特征部分 DataFrame
    test_X_scaled = preprocessing.scale(test_data[0])

    # 确保转换为 DataFrame (TCL 内部可能需要列名或 DataFrame 格式)
    test_X_df = pd.DataFrame(test_X_scaled, columns=test_data[0].columns)

    start_time = time.perf_counter()

    try:
        # 3. 运行 TCL 算法
        # TCL 通常返回预测标签 predict_y 和阈值 metric_threshold
        predict_y, _ = TCL(test_X_df)

        # 对于无监督/阈值方法，通常将预测标签作为 score 传入 rank_measure
        pred_scores = predict_y.astype(float)

        exec_time = time.perf_counter() - start_time

        # 4. 计算指标 (对齐 32 列)
        y_true = np.array(test_label).astype(int)
        y_pred = np.array(predict_y).astype(int)

        # m1: 分类指标 (10个)
        m1 = performanceMeasure.get_measure(y_true, y_pred)

        # m2 & m3: 努力感知指标 (21个: 11个c系列 + 10个m系列)
        res_rank = rankMeasurev2.rank_measure(pred_scores, t_loc, test_label)
        m2 = res_rank[:11]
        m3 = res_rank[11:]

        # 5. 整合结果 (m1 + m2 + m3 + time = 32列)
        full_measures = list(m1) + list(m2) + list(m3) + [exec_time]

        # 6. 保存 Committee 结果
        res_dir = create_dir(os.path.join(save_path, model_name + '_results'))
        save_results(os.path.join(res_dir, project_name), full_measures)

    except Exception as e:
        print(f"  Iteration {randseed} Error in TCL: {e}")


# ------------------------------------------------------------------------------
# 2. 主程序
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 【实验配置】
    Reps = 100  # 对齐主实验的重复次数
    data_dir = '../data/'
    save_path_root = '../result_USDP/TCL/'
    model_tag = 'INTC_TCL'  # 统一命名规范

    project_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for file in project_list:
        p_name = file[:-4]
        raw_df = pd.read_csv(os.path.join(data_dir, file))

        # 1. 准备标签与 LOC
        loc_candidates = ['CountLineCode', 'loc', 'LOC']
        loc_col = next((c for c in loc_candidates if c in raw_df.columns), raw_df.columns[0])
        LOC_series = raw_df[loc_col]

        label_col = raw_df.columns[-1]
        y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)
        X_raw = raw_df.iloc[:, :-1]

        # 2. 封装数据包 [全量特征DataFrame, 标签Series] (跳过特征选择)
        X_package = [X_raw, y_series]

        print(f"\n>>> 开始运行 {model_tag} | 项目: {p_name} | 原始特征数: {X_raw.shape[1]}")

        # 3. 运行重复实验
        for r in range(Reps):
            run_tcl_iteration(X_package, LOC_series, save_path_root, p_name, model_tag, r)
            if (r + 1) % 10 == 0:
                print(f"  Progress: {r + 1}/{Reps} rounds completed.")

    print("\n✅ TCL 对比实验运行完毕！")