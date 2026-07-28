import csv
import os
import time
import warnings
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from algorithms.InterpreableClustering_V1 import InterpretableClustering
from algorithms.labelingCluster import labelCluster
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap


def scott_knott_1d_grouping(importances, feature_names):
    """
    💡【内置学术级分组函数】：专门针对 1 维特征重要性设计的变分最大类间方差分组算法。
    等等同于在 1 维方差约束下，动态寻找断层点，分离出具有统计显著统治地位的 Group 1 特征。
    """
    importances = np.array(importances).flatten()

    # 按照权重从大到小排序
    sorted_idx = np.argsort(importances)[::-1]
    sorted_imps = importances[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    # 过滤掉权重完全为 0 的特征（0 绝对不属于 Group 1）
    valid_mask = sorted_imps > 0
    if not np.any(valid_mask):
        # 如果全是 0，兜底拿前 3 个
        return sorted_names[:3]

    valid_imps = sorted_imps[valid_mask]
    valid_names = sorted_names[:len(valid_imps)]

    if len(valid_imps) <= 1:
        return list(valid_names)

    # 寻找最佳分割点，使得两组之间的类间方差最大 (类似于1维 KMeans / Otsu)
    best_variance = -1
    split_idx = 1
    total_mean = np.mean(valid_imps)

    for i in range(1, len(valid_imps)):
        group1 = valid_imps[:i]
        group2 = valid_imps[i:]

        weight1 = len(group1) / len(valid_imps)
        weight2 = len(group2) / len(valid_imps)

        # 类间方差公式
        between_variance = weight1 * (np.mean(group1) - total_mean) ** 2 + weight2 * (np.mean(group2) - total_mean) ** 2

        if between_variance > best_variance:
            best_variance = between_variance
            split_idx = i

    # 切出 Group 1 (高权重组)
    group1_features = valid_names[:split_idx]

    # 学术保护锁：防止分割点太靠后把大半特征都切了。如果 Group 1 包含超过 10 个特征，强行截取前 5 个最显性的
    if len(group1_features) > 10:
        group1_features = group1_features[:5]

    return group1_features


def run_(X_data, LOC, n_class, v_lambda, max_iters, save_path, project_name, model_name, randseed, cluster_method='sc'):
    print(f"{project_name}: -> {model_name} ({cluster_method}) Round {randseed + 1} Start!")

    # 数据切分 (Bootstrap)
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_data, randseed)
    test_LOC = LOC.iloc[test_idx].values

    # 特征处理
    feature_names = X_data[0].columns.values
    test_X = preprocessing.scale(test_data[0])
    n_feas = test_X.shape[1]

    start = time.perf_counter()

    # 实例化模型
    model = InterpretableClustering(
        n_clusters=n_class,
        hidden_dims=[n_feas * 2, n_feas],
        latent_dim=n_feas,
        lambda_ce=v_lambda,
        cluster_type=cluster_method,
        random_state=randseed
    )

    # 训练与预测
    clus_label = model.fit_predict(test_X, max_iters=max_iters)
    t = time.perf_counter() - start

    # 保存消融后的可解释性报告
    report = model.get_interpretability_report(feature_names=feature_names)
    fres_dir = create_dir(os.path.join(save_path, model_name, "reports"))
    save_results_pickle(os.path.join(fres_dir, project_name), report)

    # 评估指标计算
    predict_y = labelCluster(test_X, clus_label)

    # 接收分类与努力感知指标并保存
    m1 = performanceMeasure.get_measure(test_label, predict_y)
    res_rank = rankMeasurev2.rank_measure(predict_y, test_LOC, test_label)
    measure = list(m1) + list(res_rank) + [t]

    res_path = create_dir(os.path.join(save_path, model_name + '_results'))
    save_results(os.path.join(res_path, project_name), measure)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 【基础配置】
    methods_to_run = ['kmeans']
    Reps = 100
    n_class = 2
    v_lambda = 0.1
    max_iters = 10
    data_dir = '../data/'

    # 【消融配置】：历史全特征（Baseline）结果和报告存放的根目录
    baseline_dir = '../result_20260526_VAE/clustering/'

    for cluster_method in methods_to_run:
        # 保存路径指向专属的 SKE 消融文件夹
        current_save_path = f'../result_20260526_VAE/RQ3passroot/ablation_SKE_Group1/'
        model_name = f'INTC_{cluster_method.upper()}'

        # 定位当前算法对应的历史全特征报告 reports 文件夹路径
        baseline_report_path = os.path.join(baseline_dir, model_name, "reports")
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

            # 2. 识别标签列及特征列
            X_features = data.iloc[:, :-1]
            y = data.iloc[:, -1].copy()
            y[y > 1] = 1  # 转换为二分类

            # ============================================================
            # 🌟【自动化消融核心】：内置统计分组算法动态读取并切除 Group 1 特征
            # ============================================================
            group1_features = []
            pickle_file_path = os.path.join(baseline_report_path, project_name_base + '.pkl')

            if os.path.exists(pickle_file_path):
                try:
                    with open(pickle_file_path, 'rb') as f_reader:
                        report_data = pickle.load(f_reader)

                    if isinstance(report_data, dict) and 'feature_importances' in report_data:
                        raw_importances = report_data['feature_importances']

                        # 调用内置的变分最大类间方差检验算法进行动态分组
                        group1_features = scott_knott_1d_grouping(raw_importances, list(X_features.columns))
                    else:
                        print(f"[Warning] 项目 [{project_name_base}] 报告中未发现 feature_importances！")
                except Exception as e:
                    print(f"[Error] 统计分析项目 [{project_name_base}] 失败: {str(e)}")
            else:
                print(f"[Warning] 未找到项目 [{project_name_base}] 的历史报告: {pickle_file_path}")

            # 执行 Group 1 靶向集体切除
            current_columns_list = list(X_features.columns)
            valid_targets = [f for f in group1_features if f in current_columns_list]

            if len(valid_targets) > 0:
                print(
                    f"[-] 项目 [{project_name_base}] 通过显著性分组识别出 Group 1 特征共 {len(valid_targets)} 个: {valid_targets}")
                X_features = X_features.drop(columns=valid_targets)
            else:
                print(f"[Warning] 项目 [{project_name_base}] 未释放有效 Group 1 靶向特征，保持原样运行！")
            # ============================================================

            X_data = [X_features, y]
            print(
                f">>> 开始重训消融算法: {cluster_method} | 项目: {project_name_base} | 残留特征数: {X_features.shape[1]}")

            for loop in range(Reps):
                run_(X_data, LOC, n_class, v_lambda, max_iters,
                     current_save_path, project_name_base, model_name,
                     loop, cluster_method=cluster_method)

    print("\n🚀 基于显著性统计分组的第一组特征靶向消融实验全部运行完毕！")