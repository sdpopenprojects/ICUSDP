import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

# 自动处理路径问题
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 导入模型类和辅助工具
from algorithms.InterpreableClustering_V1_CAE import InterpretableClustering
from algorithms.labelingCluster import labelCluster
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap


# ------------------------------------------------------------------------------
# 1. 实验运行函数 (严格对齐 32 列指标体系与 VAE 实验调用链)
# ------------------------------------------------------------------------------
def run_unsupervised_cae_iteration(X_package, LOC, n_class, max_iters, save_path, project_name, model_name, randseed,
                                   cluster_method):
    """
    运行单次无监督对比自编码聚类实验，完美解包并对齐32列指标
    """
    print(f"{project_name}: -> {model_name} ({cluster_method}) Round {randseed + 1} Start!")

    # 1. 数据切分 (Bootstrap)
    train_res, train_label, test_res, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)

    # 解包数据：注意原版 INTC 逻辑中核心训练与预测都在测试集切片上完成
    test_X_df = test_res[0] if isinstance(test_res, list) else test_res

    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]

    feature_names = X_package[0].columns.values

    # 2. 预处理 (与主实验 VAE 保持严格一致的 scale 转换形式)
    test_X_scaled = preprocessing.scale(test_X_df.values)
    n_feas = test_X_scaled.shape[1]

    start_time = time.perf_counter()

    try:
        # 3. 实例化融合了“对比+AE重构”机制的新模型 (【核心修改】：完美去掉 batch_size 外部参数)
        model = InterpretableClustering(
            n_clusters=n_class,
            hidden_dims=[n_feas * 2, n_feas],
            latent_dim=n_feas,
            clf='DT',
            epochs=200,        # 完美对齐架构图推荐的 200 次迭代
            lr=1e-3,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lambda_recon=0.7,  # 完美同步论文核心超参
            random_state=randseed
        )

        # 4. 模型拟合与伪标签预测 (直接传入对齐架构图 Phase 2 演进的 cluster_method)
        test_predict_labels = model.fit_predict(test_X_scaled, max_iters=max_iters, cluster_method=cluster_method)

        # 5. 簇群标签高低风险转换逻辑 (解包后保证是一维 np.array)
        y_test_predict = np.array(test_predict_labels).flatten()
        labeled_test_cluster = labelCluster(test_X_scaled, y_test_predict)

        exec_time = time.perf_counter() - start_time

        # 6. 指标计算 (完全对齐 VAE 实验与 ONE 框架的 32 列格式)
        y_true = np.array(test_label).astype(int)
        y_pred = np.array(labeled_test_cluster).astype(int)

        # m1: 基本分类指标 (10个)
        m1 = performanceMeasure.get_measure(y_true, y_pred)

        # m2 & m3: 努力感知指标 (21个)
        res_rank = rankMeasurev2.rank_measure(y_pred.astype(float), t_loc, test_label)
        m2 = res_rank[:11]  # c系列指标 (20%代码行成本)
        m3 = res_rank[11:]  # m系列指标 (基于缺陷数)

        # 7. 严格拼装成 32 列格式：10 (m1) + 11 (m2) + 10 (m3) + 1 (时间)
        full_measures = list(m1) + list(m2) + list(m3) + [exec_time]

        # 8. 保存实验结果 (规范化命名为 {model_name}_results 目录，方便与 Baseline 直接进行统计对比)
        res_dir = create_dir(os.path.join(save_path, f"{model_name}_results"))
        save_results(os.path.join(res_dir, project_name), full_measures)

        # 9. 保存可解释性模型报告 (建立在 Latent Space Z 轴上的白盒决策树规则)
        try:
            report = model.get_report(feature_names)
            report_dir = create_dir(os.path.join(save_path, model_name, "reports"))
            save_results_pickle(os.path.join(report_dir, project_name), report)
        except Exception as report_e:
            print(f"  Warning saving report for {project_name}: {report_e}")

    except Exception as e:
        print(f"  !!! Iteration {randseed} Error in {model_name}: {e}")
        import traceback
        traceback.print_exc()


# ------------------------------------------------------------------------------
# 2. 主程序控制流
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 【对齐实验标准配置】
    methods_to_run = ['gmm', 'agglomerative', 'kmedoids']
    Reps = 100
    n_class = 2
    max_iters = 10

    data_dir = '../data/'
    # 结果保存根路径，自动划分出 CAE 的专属结果区间
    save_path_root = f'../result_20260526_CAE/clustering/'

    project_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for cluster_method in methods_to_run:
        model_name = f'INTC_{cluster_method.upper()}'
        print(f"\n{'#' * 70}\n正在评估无监督对比模型: {model_name}\n{'#' * 70}")

        for file_name in project_files:
            p_name = file_name[:-4]
            raw_df = pd.read_csv(os.path.join(data_dir, file_name))

            # 1. 动态兼容并识别代码行数 LOC 列
            if 'CountLineCode' in raw_df.columns:
                LOC_series = raw_df['CountLineCode']
            elif 'loc' in raw_df.columns:
                LOC_series = raw_df['loc']
            else:
                LOC_series = raw_df.iloc[:, 0]

            # 2. 识别软件缺陷标签，提取纯特征空间 (完美对齐 demo_INTC.py 原始切分边界)
            X_raw = raw_df.iloc[:, :-1]
            y_series = raw_df.iloc[:, -1].copy()
            y_series[y_series > 1] = 1  # 转换为标准的二分类标签

            # 封装成标准数据包送入迭代
            X_package = [X_raw, y_series]

            print(f"\n>>> 开始运行 {p_name} | 特征维度: {X_raw.shape[1]} | 迭代 {Reps} 轮...")
            for randseed in range(Reps):
                run_unsupervised_cae_iteration(
                    X_package=X_package,
                    LOC=LOC_series,
                    n_class=n_class,
                    max_iters=max_iters,
                    save_path=save_path_root,
                    project_name=p_name,
                    model_name=model_name,
                    randseed=randseed,
                    cluster_method=cluster_method
                )

    print("\n✅ 对比自编码聚类实验完成，32列指标结果已成功保存至:", save_path_root)