import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn import preprocessing
from itertools import combinations

# 自动处理路径问题
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from algorithms.Classifiers2 import OptimizingCLF
from utilities.File import create_dir
from utilities.bootstrapCV import outofsample_bootstrap


# ------------------------------------------------------------------------------
# 1. 实验运行函数 (聚焦提取可解释性核心指标)
# ------------------------------------------------------------------------------
def run_supervised_iteration(X_package, randseed, model_name):
    """
    运行单次监督学习迭代，并动态抓取特征重要性，计算特征稀疏度
    """
    # 1. 数据切分 (Bootstrap)
    train_res, train_label, test_res, test_label, _, _ = outofsample_bootstrap(X_package, randseed)

    # 解包 DataFrame/Array
    train_X_df = train_res[0] if isinstance(train_res, list) else train_res
    feature_names = X_package[0].columns.values
    n_feas = train_X_df.shape[1]

    # 2. 预处理 (使用 scale)
    train_X_scaled = preprocessing.scale(train_X_df.values)

    try:
        # 3. 超参数优化
        sub_res, sub_label, val_res, val_label, _, _ = outofsample_bootstrap(
            [pd.DataFrame(train_X_scaled), train_label], randseed)

        actual_sub_train_x = sub_res[0].values if isinstance(sub_res, list) else sub_res.values
        actual_val_x = val_res[0].values if isinstance(val_res, list) else val_res.values

        # 获取优化后的分类器
        opt_model = OptimizingCLF(actual_sub_train_x, sub_label, actual_val_x, val_label, classifier=model_name)
        clf = opt_model.getOptCLF()

        # 4. 最终训练
        clf.fit(train_X_scaled, train_label)

        # 5. 【核心提取】动态适配不同基准模型的特征重要性接口
        importances = np.zeros(n_feas)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            # 针对 LR 和 linearSVM，取绝对值作为权重
            importances = np.abs(clf.coef_[0])

        # 计算 Feature_Sparsity（使用特征占比：有贡献的特征数 / 总特征数）
        used_features = np.sum(importances > 0)
        feature_sparsity = float(used_features / n_feas) if n_feas > 0 else 0.0

        # 提取 Top 10 特征名字用于外层计算 Jaccard 稳定性
        sorted_indices = np.argsort(importances)[::-1]
        current_top_features = set([feature_names[idx] for idx in sorted_indices[:10]])

        inter_metrics = {
            'Feature_Sparsity': feature_sparsity
        }
        return inter_metrics, current_top_features

    except Exception as e:
        print(f"  Iteration {randseed} Error in {model_name}: {e}")
        # 异常兜底
        return {'Feature_Sparsity': 0.0}, set()


# ------------------------------------------------------------------------------
# 2. 主程序逻辑 (精简指标生成大汇总表)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    # 【实验配置】
    Reps = 100
    # 包含所有 6 个需要对比的监督方法
    model_names = ['DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM']

    data_dir = r'../data/'
    # 结果输出路径切换为独立的可解释性对比专用夹
    save_path_root = r'F:\ICUSDP\INTC\ICUSDP\result_SDP\supervised_interpretability'

    project_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for model_tag in model_names:
        print(f"\n{'#' * 70}\n正在评估基准模型可解释性: {model_tag}\n{'#' * 70}")
        create_dir(save_path_root)

        summary_all_projects = []

        for file in project_list:
            p_name = file[:-4]
            raw_df = pd.read_csv(os.path.join(data_dir, file))

            label_col = 'label' if 'label' in raw_df.columns else raw_df.columns[-1]
            y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)

            # 安全提取原始特征
            X_raw = raw_df.drop(columns=[label_col]) if label_col in raw_df.columns else raw_df.iloc[:, :-1]
            X_package = [X_raw, y_series]

            print(f">>> 开始运行 {model_tag} | 项目: {p_name} | 原始特征数: {X_raw.shape[1]}")

            project_feature_sets = []
            project_inter_results = []

            # 100轮 Bootstrap 循环
            for r in range(Reps):
                inter_res, top_features = run_supervised_iteration(X_package, r, model_name=model_tag)
                project_inter_results.append(inter_res)
                project_feature_sets.append(top_features)

            # 🔴 外层结算 Jaccard_Stability(AJS) 稳定性
            jaccard_scores = []
            for set_i, set_j in combinations(project_feature_sets, 2):
                union_len = len(set_i.union(set_j))
                jaccard = len(set_i.intersection(set_j)) / union_len if union_len != 0 else 0.0
                jaccard_scores.append(jaccard)

            project_ajs_score = np.mean(jaccard_scores) if len(jaccard_scores) > 0 else 1.0

            # 汇总当前项目均值
            df_project_inter = pd.DataFrame(project_inter_results)
            project_mean_dict = df_project_inter.mean().to_dict()
            project_mean_dict['Project'] = p_name
            project_mean_dict['Jaccard_Stability(AJS)'] = project_ajs_score

            summary_all_projects.append(project_mean_dict)
            print(f">> 项目 {p_name} 100轮完结！AJS = {project_ajs_score:.4f}")

        # =========================================================================
        # 🌟【精简大汇总表生成】每一个模型跑完，单独生成一份只有两项指标的报告
        # =========================================================================
        if len(summary_all_projects) > 0:
            df_summary = pd.DataFrame(summary_all_projects)

            # 严格按照需求过滤并重排字段顺序
            ordered_cols = ['Project', 'Feature_Sparsity', 'Jaccard_Stability(AJS)']
            existing_cols = [c for c in ordered_cols if c in df_summary.columns]
            df_summary = df_summary[existing_cols]

            # 追加计算 Average 均值行
            total_avg = df_summary.mean(numeric_only=True).to_dict()
            total_avg['Project'] = 'Average'
            df_summary = pd.concat([df_summary, pd.DataFrame([total_avg])], ignore_index=True)

            # 导出 CSV 汇总表格
            summary_save_file = os.path.join(save_path_root, f"Compare_interpretability_{model_tag}_summary.csv")
            df_summary.to_csv(summary_save_file, index=False)

            print(f"\n========================================================")
            print(f" ✨ 基准模型【{model_tag}】可解释性指标对照表已成功更新！")
            print(f" 📂 导出路径: {summary_save_file}")
            print(f"========================================================")

    print("\n✅ 所有监督方法的 Feature_Sparsity 和 AJS 稳定性对比实验全部运行完成！")