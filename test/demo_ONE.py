import os
import csv
import pandas as pd
import numpy as np
from utilities import performanceMeasure, rankMeasurev2
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.File import create_dir, save_results

# 导入你的 ONE 算法
from algorithms.algorithm_ONE import ONE


# 🎯【修改点 1】：将 reps 的默认参数值改为 1（只允许跑一次）
def run_ONE_on_JIRA(data_path, save_path, reps=1):
    # 创建特征名称保存目录（保持原样）
    fs_save_path = r'F:\ICUSDP\INTC\ICUSDP\result_ONE\FS_feature'
    if not os.path.exists(fs_save_path):
        os.makedirs(fs_save_path)

    project_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])

    # 保持原样：从 0.31 到 0.39 的循环
    cutoff_list = np.arange(0.31, 0.40, 0.01)

    for current_cutoff in cutoff_list:
        # 为了防止浮点数精度问题，四舍五入保留2位小数并转换为字符串作为文件夹后缀
        cutoff_str = f"{current_cutoff:.2f}"
        print(f"\n==================================================")
        print(f"🚀 开始运行 cutoff_pct = {cutoff_str} 的实验 (Single Run)...")
        print(f"==================================================")

        # 保持原样：动态为每一个独立的阈值创建专属的保存文件夹
        res_root = create_dir(os.path.join(save_path, f'ONE_Results_cutoff_{cutoff_str}'))

        for file_name in project_files:
            project_name = file_name[:-4]

            csv_path = os.path.join(data_path, file_name)
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                separator = ';' if ';' in first_line else ','

            data = pd.read_csv(csv_path, sep=separator)
            data.columns = data.columns.str.strip()

            bug_col_exact = None
            for col in data.columns:
                if col.lower() == 'bugs':
                    bug_col_exact = col
                    break

            if bug_col_exact is None:
                raise KeyError(f"在文件 {file_name} 中未找到任何形如 'bugs' 的列。")

            y_raw = data[bug_col_exact].copy()
            y_numeric = pd.to_numeric(y_raw, errors='coerce').fillna(0)
            y = np.where(y_numeric >= 1, 1, 0)

            bug_columns = [col for col in data.columns if 'bug' in col.lower()]
            exclude_cols = bug_columns + ['classname']
            X_raw = data.drop(columns=[col for col in exclude_cols if col in data.columns])
            X = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)

            X_with_y = X.copy()
            X_with_y['target_label'] = y

            if 'CountLineCode' in data.columns:
                LOC = data['CountLineCode']
            elif 'loc' in data.columns:
                LOC = data['loc']
            elif 'linesAddedUntil' in data.columns:
                LOC = data['linesAddedUntil']
            else:
                LOC = X.iloc[:, 0]

            print(f">>> Project {project_name} [cutoff={cutoff_str}]: Processing...")

            # 🎯【修改点 2】：去掉原来的 for r in range(reps) 的 30 轮循环
            # 既然单轮执行，直接传入项目的全量原始数据进行确定性计算
            t_label = y
            t_loc = LOC.values

            # 将当前的 current_cutoff 动态传入 ONE 算法
            final_data = ONE(t_loc, t_label, cutoff_pct=current_cutoff)

            final_data['score'] = final_data['predict_label']
            final_data = final_data.sort_values(by=['original_idx'], ascending=[True])

            one_scores = final_data['score'].values
            one_predict_y = final_data['predict_label'].values

            y_true_int = np.array(t_label).astype(int)
            y_pred_int = np.array(one_predict_y).astype(int)

            m1 = performanceMeasure.get_measure(y_true_int, y_pred_int)

            # 调用包含了你最新修改的 ROI = TP / PCI 版本的 rank_measure
            res_rank = rankMeasurev2.rank_measure(one_scores, t_loc, t_label)

            m2 = res_rank[:11]
            m3 = res_rank[11:]

            measure = list(m1) + list(m2) + list(m3) + [0.0]

            # 结果落盘（此时每个阈值文件夹下的项目 CSV 里天生就只有完美的 1 行数据）
            save_results(os.path.join(res_root, project_name), measure)

            print(f"Project {project_name}: Single round completed successfully for cutoff {cutoff_str}.")


if __name__ == '__main__':
    data_input_path = '../data2/'
    result_output_path = 'F:/ICUSDP/INTC/ICUSDP/result_ONE/AEEEM/'

    # 🎯【修改点 3】：传参明确指定为 1
    run_ONE_on_JIRA(data_input_path, result_output_path, reps=1)