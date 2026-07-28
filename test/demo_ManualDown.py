import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap

# 自动处理路径问题：确保能从根目录导入算法和工具包
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def run_md(X_data, LOC, save_path, project_name, model_name, randseed):
    """
    运行 ManualDown (MD) 算法
    MD逻辑：LOC越大，缺陷概率越高（简单的努力感知基准模型）
    """
    # 1. 数据切分 (Bootstrap)
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_data, randseed)

    # 确保获取测试集的LOC
    if isinstance(LOC, pd.Series):
        test_LOC = LOC.iloc[test_idx].values
    else:
        test_LOC = LOC[test_idx]

    # 特征处理：MD 实际上不直接使用 X 训练，但为了对齐实验流程，我们依然保留 scale 步骤
    test_X = preprocessing.scale(test_data[0])

    start = time.perf_counter()

    # --- MD 算法核心逻辑 ---
    # ManualDown 预测得分就是 LOC 本身（或者负 LOC，取决于具体实现，通常 LOC 大 = 风险高）
    # 这里直接使用 test_LOC 作为预测得分
    score = test_LOC.copy()

    t = time.perf_counter() - start

    # --- 评估指标计算 ---
    # 判定阈值：通常取中位数，前50%的代码量模块定为有缺陷
    predict_y = np.where(score >= np.median(score), 1, 0)

    y_true_int = np.array(test_label).astype(int)
    y_pred_int = np.array(predict_y).astype(int)

    # 1. 分类指标 (m1: 10个)
    m1 = performanceMeasure.get_measure(y_true_int, y_pred_int)

    # 2. 排序与模块维度指标 (21个)
    # 返回：Popt, c系列(10个), m系列(10个)
    res_rank = rankMeasurev2.rank_measure(score, test_LOC, test_label)

    # 拆分为 m2 (c系列, 11个) 和 m3 (m系列, 10个)
    m2 = res_rank[:11]
    m3 = res_rank[11:]

    # 3. 整合所有指标 (32列)
    measure = list(m1) + list(m2) + list(m3) + [t]

    # 保存结果
    res_path = create_dir(os.path.join(save_path, model_name + '_results'))
    save_results(os.path.join(res_path, project_name), measure)


if __name__ == '__main__':
    # 屏蔽 FutureWarning 警告
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore')

    # 【配置项】
    Reps = 100
    data_dir = '../data/'
    # 保存路径
    current_save_path = f'../result_USDP/MD/'
    model_name = 'INTC_MD'

    project_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for file_name in project_files:
        project_name_base = file_name[:-4]
        data = pd.read_csv(os.path.join(data_dir, file_name))

        # 1. 识别 LOC 列
        if 'CountLineCode' in data.columns:
            LOC = data['CountLineCode']
        elif 'loc' in data.columns:
            LOC = data['loc']
        else:
            LOC = data.iloc[:, 0]

        # 2. 准备特征和标签
        X_features = data.iloc[:, :-1]
        y = data.iloc[:, -1].copy()
        y[y > 1] = 1  # 二值化

        # 3. 封装为列表格式供 bootstrap 使用 (跳过特征选择，直接传递完整的 X_features)
        X_data = [X_features, y]

        print(f"\n>>> 正在运行 MD 算法 | 项目: {project_name_base} | 原始特征数: {X_features.shape[1]}")

        # 运行 Reps 轮实验
        for loop in range(Reps):
            run_md(X_data, LOC, current_save_path, project_name_base, model_name, loop)
            if (loop + 1) % 10 == 0:
                print(f"  {project_name_base}: Round {loop + 1} completed.")

    print("\n所有 JIRA 项目 MD 算法运行完毕！")