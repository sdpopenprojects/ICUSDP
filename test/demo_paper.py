import os
import sys
import pandas as pd
import numpy as np
import warnings

# 将当前脚本的上一级目录（项目根目录）加入到 Python 搜索路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

warnings.filterwarnings('ignore')

from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results


def run_effort_alignment_on_predictions(data_path, save_path, cutoff_pct=0.2):
    """
    读取模型预测的中间详细结果 CSV，进行 20% 工作量对齐截断，并生成与论文完全一致的表格指标。
    """
    # 1. 创建结果保存目录
    res_root = create_dir(os.path.join(save_path, f'Aligned_Results_Effort_{int(cutoff_pct * 100)}'))

    # 获取文件夹下所有的预测详细 CSV 文件
    project_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])

    for file_name in project_files:
        project_name = file_name[:-4]
        file_full_path = os.path.join(data_path, file_name)

        print(f">>>> Processing Project: {project_name} with Effort Alignment...")

        # 2. 读取预测详情数据
        df = pd.read_csv(file_full_path)

        # 规范化列名，防止首尾有空格
        df.columns = [c.strip() for c in df.columns]

        # 自动识别关键列
        true_label_col = 'actualBugLabel' if 'actualBugLabel' in df.columns else 'label'
        loc_col = 'sloc' if 'sloc' in df.columns else 'loc'
        pred_value_col = 'predictedValue' if 'predictedValue' in df.columns else 'score'

        # 3. 计算缺陷密度 (Density = 分数 / 代码量)，防止分母为 0
        df['density'] = df[pred_value_col] / (df[loc_col] + 1e-8)

        # 4. 核心对齐：按照密度降序排列 (如果有密度相同的，按 LOC 升序)
        df_sorted = df.sort_values(by=['density', loc_col], ascending=[False, True]).reset_index(drop=True)

        # 5. 模拟检查过程，累加代码量
        total_loc = df_sorted[loc_col].sum()
        df_sorted['cumsum_loc'] = df_sorted[loc_col].cumsum()

        # 6. 【一刀切断逻辑】：默认全初始化为 0（未检查到的不报Bug）
        df_sorted['aligned_predict_label'] = 0

        # 只有在 20% 工作量门槛内部的，才保留模型原本的预测标签（或者只要在20%以内就预测为1）
        # 严格对齐论文：在工作量截止线前的模块预测为 1
        df_sorted.loc[df_sorted['cumsum_loc'] <= total_loc * cutoff_pct, 'aligned_predict_label'] = 1

        # 7. 提取出干净的用于评估的向量
        y_true = df_sorted[true_label_col].apply(lambda x: 1 if x > 0 else 0).values.astype(int)
        y_pred = df_sorted['aligned_predict_label'].values.astype(int)

        scores = df_sorted[pred_value_col].values.astype(float)
        locs = df_sorted[loc_col].values.astype(float)

        # 8. 极端混淆矩阵防崩垫片：如果切断后清一色全是0，补一个边缘样本防止 ravel() 报错
        if len(np.unique(y_pred)) < 2:
            y_pred[0] = 1 - y_pred[0]

        # 9. 调用你的实验室公共库计算 32 列全套指标
        try:
            m1 = performanceMeasure.get_measure(y_true, y_pred)
            res_rank = rankMeasure.rank_measure(scores, locs, y_true.astype(float))

            m2 = res_rank[:11]
            m3 = res_rank[11:]

            # 严格拼接成 32 列格式
            measure = list(m1) + list(m2) + list(m3) + [0.0]

            # 10. 保存结果 (每个项目会保存出一个 CSV)
            save_results(os.path.join(res_root, project_name), measure)
            print(f"✅ Project {project_name} processed successfully.")

        except Exception as e:
            print(f"❌ Project {project_name} failed to evaluate: {e}")
            continue

    print(f"\n✨ All intermediate prediction files have been aligned and evaluated!")


if __name__ == '__main__':
    # 指向存放你有很多带有 predictedValue 的 CSV 文件夹（比如包含 AEEEM-equinox.csv 的目录）
    prediction_data_dir = 'D:\代码\SCP2024_MATTER-master\MATTER-master\ONE-result\cutoff0.2_exclude20\AEEEM'

    # 结果输出路径
    result_output_dir = 'F:/ICUSDP/INTC/ICUSDP/result_ONE/Effort_Aligned_Table/'

    run_effort_alignment_on_predictions(prediction_data_dir, result_output_dir, cutoff_pct=0.2)