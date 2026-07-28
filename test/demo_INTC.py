import csv
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from algorithms.InterpreableClustering_V1 import InterpretableClustering
from algorithms.labelingCluster import labelCluster
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X_data, LOC, n_class, v_lambda, max_iters, save_path, project_name, model_name, randseed, cluster_method='sc'):
    print(f"{project_name}: -> {model_name} ({cluster_method}) Round {randseed + 1} Start!")

    # 数据切分 (Bootstrap)
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_data, randseed)
    test_LOC = LOC.iloc[test_idx].values  # 确保 LOC 获取方式与 test_idx 匹配

    # 特征处理
    # 注意：X_data[0] 是 DataFrame
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

    # 保存可解释性报告
    report = model.get_interpretability_report(feature_names=feature_names)
    fres_dir = create_dir(os.path.join(save_path, model_name, "reports"))
    save_results_pickle(os.path.join(fres_dir, project_name), report)

    # 评估指标计算
    predict_y = labelCluster(test_X, clus_label)

    # 1. 接收 10 个分类指标 (m1)
    m1 = performanceMeasure.get_measure(test_label, predict_y)

    # 2. 接收 21 个努力感知、模块维度指标
    # 返回顺序：Popt, c系列(10个), m系列(10个)
    res_rank = rankMeasurev2.rank_measure(predict_y, test_LOC, test_label)

    # 拆分为 m2 (c系列, 11个) 和 m3 (m系列, 10个)
    m2 = res_rank[:11]
    m3 = res_rank[11:]

    # 3. 整合所有指标向量 32 列
    # 10 (m1) + 11 (m2) + 10 (m3) + 1 (时间) = 32 列
    measure = list(m1) + list(m2) + list(m3) + [t]

    # 保存 Reps 实验结果
    res_path = create_dir(os.path.join(save_path, model_name + '_results'))
    save_results(os.path.join(res_path, project_name), measure)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 【配置项】
    methods_to_run = ['kmeans']
    # methods_to_run = ['kmeans', 'sc', 'gmm', 'agglomerative', 'kmedoids']
    Reps = 100
    n_class = 2
    v_lambda = 0.1
    max_iters = 10
    data_dir = '../data/'

    for cluster_method in methods_to_run:
        # 修改保存路径，以区分是否进行了特征选择
        current_save_path = f'../result_20260526_VAE/clustering/'
        model_name = f'INTC_{cluster_method.upper()}'

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
            y[y > 1] = 1  # 转换为二分类标签

            X_data = [X_features, y]

            print(f"\n>>> 正在运行算法: {cluster_method} | 项目: {project_name_base} | 特征数: {X_features.shape[1]}")

            for loop in range(Reps):
                run_(X_data, LOC, n_class, v_lambda, max_iters,
                     current_save_path, project_name_base, model_name,
                     loop, cluster_method=cluster_method)

    print("\n所有算法及项目运行完毕！")