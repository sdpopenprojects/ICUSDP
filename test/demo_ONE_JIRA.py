import os
import csv
import pandas as pd
import numpy as np
from utilities import performanceMeasure, rankMeasurev2
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.File import create_dir, save_results

# 导入你的 ONE 算法
from algorithms.algorithm_ONE1 import ONE


def run_ONE_on_JIRA(data_path, save_path, reps=100):
    # 1. 创建结果保存目录（🎯修改：文件夹命名为 JIRA_Results 方便区分）
    res_root = create_dir(os.path.join(save_path, 'ONE_Results'))

    # 2. 创建特征名称保存目录
    fs_save_path = r'F:\ICUSDP\INTC\ICUSDP\result_ONE\FS_feature'
    if not os.path.exists(fs_save_path):
        os.makedirs(fs_save_path)

    project_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])

    for file_name in project_files:
        project_name = file_name[:-4]

        # 兼容分号分隔符，自动去除列名前后的空格
        csv_path = os.path.join(data_path, file_name)
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            separator = ';' if ';' in first_line else ','

        data = pd.read_csv(csv_path, sep=separator)
        data.columns = data.columns.str.strip()

        # 🎯【核心修改 1：不区分大小写，自适应匹配 JIRA 的 'label' 列或原 'bugs' 列】
        bug_col_exact = None
        for col in data.columns:
            if col.lower() in ['label', 'bugs']:  # 加上对 'label' 的兼容
                bug_col_exact = col
                break

        if bug_col_exact is None:
            raise KeyError(f"在文件 {file_name} 中未找到任何形如 'label' 或 'bugs' 的列。")

        y_raw = data[bug_col_exact].copy()
        y_numeric = pd.to_numeric(y_raw, errors='coerce').fillna(0)
        y = np.where(y_numeric >= 1, 1, 0)

        # 🎯【核心修改 2：过滤特征时，清除所有包含 'label' 或 'bug' 的干扰列】
        bug_columns = [col for col in data.columns if 'bug' in col.lower() or 'label' in col.lower()]
        exclude_cols = bug_columns + ['classname']
        X_raw = data.drop(columns=[col for col in exclude_cols if col in data.columns])
        X = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)

        X_with_y = X.copy()
        X_with_y['target_label'] = y

        # 识别 LOC（🎯无需人工修改：你原来的逻辑会首先尝试找 'CountLineCode'，完美支持 JIRA）
        if 'CountLineCode' in data.columns:
            LOC = data['CountLineCode']
        elif 'CountLine' in data.columns:  # 保底增加对 CountLine 的识别
            LOC = data['CountLine']
        elif 'loc' in data.columns:
            LOC = data['loc']
        elif 'linesAddedUntil' in data.columns:
            LOC = data['linesAddedUntil']
        else:
            LOC = X.iloc[:, 0]

        print(f">>> Project {project_name}: Processing JIRA dataset without Feature Selection...")

        for r in range(reps):
            # 3. Bootstrap 切分
            _, _, _, test_label, _, test_idx = outofsample_bootstrap(X_with_y, r)

            t_label = test_label.values if hasattr(test_label, 'values') else test_label
            t_loc = LOC.iloc[test_idx].values

            # 4. 运行 ONE 算法
            final_data = ONE(t_loc, t_label)

            # =========================================================
            # 【优雅对接】直接使用 ONE 内部预测的 predict_label 作为分数值即可
            # 我们在新版 rank_measure 内部会自动完成对齐论文 R 语言的多级稳定排序
            # =========================================================
            final_data['score'] = final_data['predict_label']

            # 将预测结果恢复回测试集的原始乱序状态，交给 rank_measure 内部去按照论文规则排序
            final_data = final_data.sort_values(by=['original_idx'], ascending=[True])

            one_scores = final_data['score'].values
            one_predict_y = final_data['predict_label'].values

            # 类型转换
            y_true_int = np.array(t_label).astype(int)
            y_pred_int = np.array(one_predict_y).astype(int)

            # --- 5. 调用评估函数 ---
            m1 = performanceMeasure.get_measure(y_true_int, y_pred_int)

            # 这里的 rank_measure 已经在使用我们上一轮修改完成的、100% 对齐 R 语言的版本
            res_rank = rankMeasurev2.rank_measure(one_scores, t_loc, t_label)

            # m2: 基于 20% 代码行的指标 (内部 ROI 已经乘以 100 变成百分比)
            m2 = res_rank[:11]

            # m3: 基于 20% 模块数的指标 (内部 ROI 已经乘以 100 变成百分比)
            m3 = res_rank[11:]

            # --- 6. 整合指标 ---
            measure = list(m1) + list(m2) + list(m3) + [0.0]

            # 7. 保存结果
            save_results(os.path.join(res_root, project_name), measure)

        print(f"Project {project_name}: All {reps} rounds completed.")


if __name__ == '__main__':
    # 🎯【核心修改 3：更新 JIRA 数据集的本地输入和输出根目录路径】
    data_input_path = '../data/'  # 改为存放 JIRA 28个项目CSV的本地路径
    result_output_path = 'F:/ICUSDP/INTC/ICUSDP/result_ONE/JIRA/'  # 结果输出至 JIRA 专用夹

    run_ONE_on_JIRA(data_input_path, result_output_path, reps=100)