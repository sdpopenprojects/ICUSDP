import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn import preprocessing

# 自动处理路径问题
current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from algorithms.Classifiers2 import OptimizingCLF
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap


# ------------------------------------------------------------------------------
# 1. 实验运行函数 (调用 Classifiers2 原生报告函数，完美对齐)
# ------------------------------------------------------------------------------
def run_supervised_iteration(X_package, LOC, save_path, project_name, model_name, randseed):
    """
    运行单次监督学习迭代，解决了数据嵌套导致的 dim 3 报错，并保存可解释性报告
    """
    # 1. 数据切分 (Bootstrap)
    train_res, train_label, test_res, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)

    # 从返回的 list 中解包出真正的 DataFrame/Array
    train_X_df = train_res[0] if isinstance(train_res, list) else train_res
    test_X_df = test_res[0] if isinstance(test_res, list) else test_res

    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]

    feature_names = X_package[0].columns.values

    # 2. 预处理 (与主实验对齐：使用 scale)
    # 转换为 values 确保是 2D Numpy 数组
    train_X_scaled = preprocessing.scale(train_X_df.values)
    test_X_scaled = preprocessing.scale(test_X_df.values)

    start_time = time.perf_counter()

    try:
        # 3. 超参数优化
        # 划分验证集时，确保输入格式正确
        sub_res, sub_label, val_res, val_label, _, _ = outofsample_bootstrap(
            [pd.DataFrame(train_X_scaled), train_label], randseed)

        # 提取验证阶段的二维数据
        actual_sub_train_x = sub_res[0].values if isinstance(sub_res, list) else sub_res.values
        actual_val_x = val_res[0].values if isinstance(val_res, list) else val_res.values

        # 获取优化后的分类器
        opt_model = OptimizingCLF(actual_sub_train_x, sub_label, actual_val_x, val_label, classifier=model_name)
        clf = opt_model.getOptCLF()

        # 4. 最终训练与预测
        clf.fit(train_X_scaled, train_label)
        predict_y = clf.predict(test_X_scaled)

        # 预测分数 (用于努力感知指标排序)
        if hasattr(clf, "predict_proba"):
            pred_scores = clf.predict_proba(test_X_scaled)[:, 1]
        else:
            pred_scores = predict_y.astype(float)

        exec_time = time.perf_counter() - start_time

        # ======================================================================
        # 【新增：直接调用 Classifiers2 内部的 get_interpretability_report】
        # ======================================================================
        try:
            # 将最终训练好的、真正拥有特征重要性的模型实体赋给 opt_model
            opt_model.clfmodel = clf

            # 直接调用 Classifiers2 内部写好的精细化报告生成逻辑（包含了 DT 的 list 转换和安全机制）
            report = opt_model.get_interpretability_report(feature_names=feature_names)

            # 创建报告目录并保存为 pickle (完全对齐主实验 demo_INTC 结构：reports/project_name)
            fres_dir = create_dir(os.path.join(save_path, model_name, "reports"))
            save_results_pickle(os.path.join(fres_dir, project_name), report)

        except Exception as report_e:
            print(f"  Warning: Failed to generate interpretability report for {model_name}: {report_e}")
        # ======================================================================

        # 5. 指标计算 (严格对齐 32 列结构)
        y_true = np.array(test_label).astype(int)
        y_pred = np.array(predict_y).astype(int)

        # m1: 分类指标 (10个)
        m1 = performanceMeasure.get_measure(y_true, y_pred)

        # m2 & m3: 努力感知指标 (21个)
        res_rank = rankMeasurev2.rank_measure(pred_scores, t_loc, test_label)
        m2 = res_rank[:11]  # c系列指标
        m3 = res_rank[11:]  # m系列指标

        # 6. 合并结果 (10 + 11 + 10 + 1 = 32列)
        full_measures = list(m1) + list(m2) + list(m3) + [exec_time]

        # 7. 保存结果 (_results 文件夹隔离，完全对齐 demo_INTC)
        res_dir = create_dir(os.path.join(save_path, model_name + '_results'))
        # save_results(os.path.join(res_dir, f"{project_name}.csv"), full_measures)
        # 去掉 .csv 后缀，因为你的通用工具类 save_results 会自动补上它
        save_results(os.path.join(res_dir, project_name), full_measures)

    except Exception as e:
        print(f"  Iteration {randseed} Error in {model_name}: {e}")
        import traceback
        traceback.print_exc()


# ------------------------------------------------------------------------------
# 2. 主程序逻辑
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    # 【实验配置】MU
    Reps = 100
    # model_names = ['DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM']
    model_names = ['RF', 'GBM', 'XGBoost', 'LR', 'linearSVM']
    data_dir = r'../data/'
    save_path_root = r'F:\ICUSDP\INTC\ICUSDP\result_SDP\supervised'

    project_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for model_tag in model_names:
        print(f"\n{'#' * 70}\n正在评估基准模型: {model_tag}\n{'#' * 70}")

        for file in project_list:
            p_name = file[:-4]
            raw_df = pd.read_csv(os.path.join(data_dir, file))

            # 1. 准备标签与 LOC (优先匹配 JIRA 数据集列名)
            loc_candidates = ['CountLineCode', 'loc', 'LOC']
            loc_col = next((c for c in loc_candidates if c in raw_df.columns), raw_df.columns[0])
            LOC_series = raw_df[loc_col]

            label_col = 'label' if 'label' in raw_df.columns else raw_df.columns[-1]
            y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)

            # 安全提取原始特征 (确保去除标签列，防止引发数据泄漏)
            X_raw = raw_df.drop(columns=[label_col]) if label_col in raw_df.columns else raw_df.iloc[:, :-1]

            # 2. 封装数据包 (跳过特征选择，直接透传全量原始特征 X_raw)
            X_package = [X_raw, y_series]

            # 3. 重复实验
            print(f">>> 开始运行 {model_tag} | 项目: {p_name} | 原始特征数: {X_raw.shape[1]}")
            for r in range(Reps):
                run_supervised_iteration(X_package, LOC_series, save_path_root, p_name, model_tag, r)
                if (r + 1) % 10 == 0:
                    print(f"  {p_name} ({model_tag}): {r + 1}/{Reps} 完成")

    print("\n✅ 所有监督学习对齐实验运行完成！")