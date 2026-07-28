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

# 导入修改好的 InterpretableClustering 类和辅助工具
from algorithms.InterpreableClustering_V1_CAE import InterpretableClustering
from algorithms.labelingCluster import labelCluster
from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.AutoSpearman import AutoSpearman


# ------------------------------------------------------------------------------
# 1. 实验运行函数 (严格对齐 32 列指标体系)
# ------------------------------------------------------------------------------
def run_unsupervised_cae_iteration(X_package, LOC, n_class, max_iters, save_path, project_name, model_name, randseed,
                                   cluster_method):
    """
    运行单次无监督对比自编码聚类实验，完美解包并对齐32列指标
    """
    # 1. 数据切分 (Bootstrap)
    train_res, train_label, test_res, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)

    # 【核心修复】：解包并清洗为标准的二维数据
    train_X_df = train_res[0] if isinstance(train_res, list) else train_res
    test_X_df = test_res[0] if isinstance(test_res, list) else test_res

    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]

    feature_names = X_package[0].columns.values

    # 2. 预处理 (使用与主实验完全一致的 scale 转换)
    train_X_scaled = preprocessing.scale(train_X_df.values)
    test_X_scaled = preprocessing.scale(test_X_df.values)
    n_feas = train_X_scaled.shape[1]

    start_time = time.perf_counter()

    try:
        # 3. 实例化融合了“对比+AE重构”机制的新模型
        model = InterpretableClustering(
            n_clusters=n_class,
            hidden_dims=[n_feas * 2, n_feas],
            latent_dim=n_feas,
            clf='DT',
            epochs=30,  # 针对 JIRA 推荐迭代 30 次
            lr=1e-3,
            batch_size=64,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lambda_recon=0.7  # 完美同步论文核心超参
        )

        # 4. 模型拟合与伪标签预测
        train_pseudo_labels = model.fit_predict(train_X_scaled, max_iters=max_iters, cluster_method=cluster_method)
        test_predict_labels = model.predict(test_X_scaled)

        # 5. 簇群标签高低风险转换逻辑 (解包后必须保证是一维 np.array)
        y_train_pseudo = np.array(train_pseudo_labels).flatten()
        y_test_predict = np.array(test_predict_labels).flatten()

        labeled_train_cluster = labelCluster(train_X_scaled, y_train_pseudo)
        labeled_test_cluster = labelCluster(test_X_scaled, y_test_predict)

        exec_time = time.perf_counter() - start_time

        # 6. 指标计算 (完全对齐主实验与 ONE 框架的 32 列格式)
        y_true = np.array(test_label).astype(int)
        y_pred = np.array(labeled_test_cluster).astype(int)

        # m1: 基本分类指标 (10个)
        m1 = performanceMeasure.get_measure(y_true, y_pred)

        # m2 & m3: 努力感知指标 (21个)
        # 注意：使用伪标签转换后的二分类预测作为排序依据
        res_rank = rankMeasure.rank_measure(y_pred.astype(float), t_loc, test_label)
        m2 = res_rank[:11]  # c系列指标 (20%代码行成本)
        m3 = res_rank[11:]  # m系列指标 (基于缺陷数)

        # 7. 严格拼装成 32 列格式
        full_measures = list(m1) + list(m2) + list(m3) + [exec_time]

        # 8. 保存实验结果
        res_dir = create_dir(os.path.join(save_path, model_name))
        save_results(os.path.join(res_dir, f"{project_name}.csv"), full_measures)

        # 保存可解释性模型报告
        try:
            report = model.get_report(feature_names)
            save_results_pickle(os.path.join(res_dir, f"{project_name}_report.pkl"), report)
        except Exception:
            pass  # 防止某些边缘情况下报告生成失败影响主流程

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
    # methods_to_run = ['kmeans', 'sc']
    methods_to_run = ['kmeans', 'sc', 'gmm', 'agglomerative', 'kmedoids']
    Reps = 30  # 修改为30次重复，与主实验及监督学习基准对齐
    n_class = 2
    max_iters = 10
    data_dir = r'F:\ICUSDP\INTC\ICUSDP\data'
    save_path_root = r'F:\ICUSDP\INTC\ICUSDP\result_20260519\CAE_FS'

    # 特征选择保存路径
    fs_record_path = os.path.join(save_path_root, 'FS_feature')
    if not os.path.exists(fs_record_path):
        os.makedirs(fs_record_path)

    project_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for cluster_method in methods_to_run:
        model_name = f'INTC_{cluster_method.upper()}'
        print(f"\n{'#' * 70}\n正在评估无监督对比模型: {model_name}\n{'#' * 70}")

        for file_name in project_files:
            p_name = file_name[:-4]
            raw_df = pd.read_csv(os.path.join(data_dir, file_name))

            # 1. 动态兼容并识别代码行数 LOC 列
            loc_candidates = ['CountLineCode', 'loc', 'LOC']
            loc_col = next((c for c in loc_candidates if c in raw_df.columns), raw_df.columns[0])
            LOC_series = raw_df[loc_col]

            # 2. 动态兼容并识别软件缺陷标签
            label_col = 'label' if 'label' in raw_df.columns else (
                'Bug' if 'Bug' in raw_df.columns else raw_df.columns[-1])
            y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)

            # 提取纯特征空间
            X_raw = raw_df.drop(columns=[label_col]) if label_col in raw_df.columns else raw_df.iloc[:, :-1]
            if loc_col in X_raw.columns:
                X_raw = X_raw.drop(columns=[loc_col])

            # 3. 核心步骤：引入 AutoSpearman 特征选择，确保与主实验特征空间完全一致
            print(f">>> 项目: {p_name} | 执行 AutoSpearman...")
            X_selected = AutoSpearman(X_raw)

            # 记录保存留档被选中的特征
            feat_names = X_selected.columns.tolist()
            with open(os.path.join(fs_record_path, f"{p_name}_features_{model_name}.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Selected_Features'])
                for fn in feat_names:
                    writer.writerow([fn])

            X_package = [X_selected, y_series]

            # 4. 执行多轮独立重复实验
            print(f">>> 开始运行 {Reps} 轮迭代...")
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
                if (randseed + 1) % 10 == 0:
                    print(f"  进度: {p_name} ({model_name}) [{randseed + 1}/{Reps}]")

    print("\n✅ 对比自编码聚类实验完成，32列指标结果已成功保存至:", save_path_root)