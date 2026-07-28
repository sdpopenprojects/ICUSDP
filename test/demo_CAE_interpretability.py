import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations

# 自动处理路径问题
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 导入模型类和辅助工具
from algorithms.InterpreableClustering_V1_CAE import InterpretableClustering
from utilities.File import create_dir
from utilities.bootstrapCV import outofsample_bootstrap


# ------------------------------------------------------------------------------
# 1. 实验运行函数 (严格 100% 维持你原汁原味的输入参数与解包逻辑)
# ------------------------------------------------------------------------------
def run_unsupervised_cae_iteration(X_package, LOC, n_class, max_iters, save_path, project_name, model_name, randseed,
                                   cluster_method):
    """
    运行单次无监督对比自编码聚类实验，完美适配底层的传参，并修正隐空间特征重要性错位问题
    """
    # 1. 数据切分 (Bootstrap)
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)

    # 2. 特征处理
    feature_names = X_package[0].columns.values
    test_X = preprocessing.scale(test_data[0])
    n_feas = test_X.shape[1]

    # 3. 实例化模型 (【彻底修复】根据源码构造函数，完全去除不支持的 cluster_type 参数)
    model = InterpretableClustering(
        n_clusters=n_class,
        hidden_dims=[n_feas * 2, n_feas],
        latent_dim=n_feas,  # 隐空间维度与原始特征数对齐 (65)
        random_state=randseed
    )

    # 4. 训练与预测 (聚类方法作为参数传入 fit_predict 函数中)
    clus_label = model.fit_predict(test_X, max_iters=max_iters, cluster_method=cluster_method)

    # 5. 动态提取可解释性指标与 Top-10 特征
    inter_metrics = {}
    current_top_features = set()

    # 为了确保拿到的是【原始特征维度 (65)】的解释性而非【隐空间维度 (32)】的解
    # 建立一个基于全量原始特征对伪标签的代理树模型
    surrogate_tree = DecisionTreeClassifier(random_state=42)
    surrogate_tree.fit(test_X, clus_label)

    importances = surrogate_tree.feature_importances_
    used_features = np.sum(importances > 0)

    # 计算原始特征的 Feature_Sparsity
    inter_metrics['Feature_Sparsity'] = float(used_features / n_feas)

    # 排序并提取真正的原始 Top-10 特征名字，用于精确计算稳定性 AJS
    sorted_indices = np.argsort(importances)[::-1]
    current_top_features = set([feature_names[idx] for idx in sorted_indices[:10]])

    return inter_metrics, current_top_features


# ------------------------------------------------------------------------------
# 2. 主程序逻辑 (在你的原版 main 骨架上只做精简汇总)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # 100% 维持你原本的配置变量名
    methods_to_run = ['kmeans']
    Reps = 100
    n_class = 2
    max_iters = 10

    data_dir = r'../data/'
    # 独立的可解释性对比专用夹
    current_save_path = r'F:\ICUSDP\INTC\ICUSDP\result_CAE\cae_interpretability'

    project_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for cluster_method in methods_to_run:
        create_dir(current_save_path)
        model_name = f'INTC_{cluster_method.upper()}'
        print(f"\n{'#' * 70}\n正在评估无监督对比模型可解释性: {model_name}\n{'#' * 70}")

        summary_all_projects = []

        for file_name in project_files:
            p_name = file_name[:-4]
            raw_df = pd.read_csv(os.path.join(data_dir, file_name))

            # 1. 动态兼容并识别代码行数 LOC 列 (100% 维持原样)
            if 'CountLineCode' in raw_df.columns:
                LOC_series = raw_df['CountLineCode']
            elif 'loc' in raw_df.columns:
                LOC_series = raw_df['loc']
            else:
                LOC_series = raw_df.iloc[:, 0]

            # 2. 识别软件缺陷标签，提取纯特征空间 (100% 维持原样)
            X_raw = raw_df.iloc[:, :-1]
            y_series = raw_df.iloc[:, -1].copy()
            y_series[y_series > 1] = 1

            # 封装成标准数据包送入迭代
            X_package = [X_raw, y_series]

            print(f"\n>>> 开始运行 {p_name} | 特征维度: {X_raw.shape[1]} | 迭代 {Reps} 轮...")

            project_feature_sets = []
            project_inter_results = []

            for loop in range(Reps):
                # 调用时完美匹配你原函数的全量位置参数
                inter_res, top_features = run_unsupervised_cae_iteration(
                    X_package, LOC_series, n_class, max_iters,
                    current_save_path, p_name, model_name,
                    loop, cluster_method=cluster_method
                )
                project_inter_results.append(inter_res)
                project_feature_sets.append(top_features)

            # 外层结算 AJS 稳定性 (交叉组合比对)
            jaccard_scores = []
            for set_i, set_j in combinations(project_feature_sets, 2):
                union_len = len(set_i.union(set_j))
                jaccard = len(set_i.intersection(set_j)) / union_len if union_len != 0 else 0.0
                jaccard_scores.append(jaccard)

            project_ajs_score = np.mean(jaccard_scores) if len(jaccard_scores) > 0 else 1.0

            df_project_inter = pd.DataFrame(project_inter_results)
            project_mean_dict = df_project_inter.mean().to_dict()
            project_mean_dict['Project'] = p_name
            project_mean_dict['Jaccard_Stability(AJS)'] = project_ajs_score

            summary_all_projects.append(project_mean_dict)
            print(f">> 项目 {p_name} 100轮完结！AJS = {project_ajs_score:.4f}")

        # =========================================================================
        # 🌟【大汇总表生成】精简过滤，只保留表格所需的两项指标
        # =========================================================================
        if len(summary_all_projects) > 0:
            df_summary = pd.DataFrame(summary_all_projects)

            # 严格按照需求过滤并重排字段顺序
            ordered_cols = ['Project', 'Feature_Sparsity', 'Jaccard_Stability(AJS)']
            existing_cols = [c for c in ordered_cols if c in df_summary.columns]
            df_summary = df_summary[existing_cols]

            # 计算 Average 行
            total_avg = df_summary.mean(numeric_only=True).to_dict()
            total_avg['Project'] = 'Average'
            df_summary = pd.concat([df_summary, pd.DataFrame([total_avg])], ignore_index=True)

            summary_save_file = os.path.join(current_save_path, f"Compare_interpretability_{model_name}_summary.csv")
            df_summary.to_csv(summary_save_file, index=False)

            print(f"\n========================================================")
            print(f" ✨ 【{model_name}】精简版可解释性对照表已成功导出！")
            print(f" 📂 汇总文件路径: {summary_save_file}")
            print(f"========================================================")

    print("\n所有算法及项目运行完毕！")