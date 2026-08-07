import os
import numpy as np
import pandas as pd
from pandas import DataFrame

if __name__ == '__main__':
    # 1. 基础路径设置
    path = r"F:\ICUSDP\INTC\ICUSDP\result_ONE\AEEEM"

    # 自动扫描 path 下所有符合条件的小文件夹
    all_sub_dirs = sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.startswith('ONE_Results_cutoff_')
    ])

    if not all_sub_dirs:
        print(f"❌ 未在路径 {path} 下找到任何以 'ONE_Results_cutoff_' 开头的文件夹！")
        all_sub_dirs = ['ONE_Results']

    # 🎯 用字典来按“项目名”分门别类地收集所有小文件夹里的数据
    # 结构如：{'equinox': [df1_values, df2_values...], 'pde': [...]}
    project_data_dict = {}

    print("🔄 开始跨文件夹读取所有项目的实验数据...")
    for model_name in all_sub_dirs:
        model_full_path = os.path.join(path, model_name)

        if not os.path.exists(model_full_path):
            continue

        files = [f for f in sorted(os.listdir(model_full_path)) if f.endswith('.csv')]

        for file in files:
            file_path = os.path.join(model_full_path, file)
            project_name = file[:-4]  # 提取项目名 (如 equinox)

            df = pd.read_csv(file_path, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')

            # 如果字典里还没有这个项目，就初始化一个列表
            if project_name not in project_data_dict:
                project_data_dict[project_name] = []

            # 把当前小文件夹下，该项目的 30 轮数据矩阵追加进来
            project_data_dict[project_name].append(df.values)

    # 如果什么都没读到，直接退出
    if not project_data_dict:
        print("❌ 未读取到任何合法的项目 CSV 数据，请检查路径。")
        exit()

    files_list = []  # 用于记录最终表格的行索引 (项目名)
    all_global_rows = []  # 用于记录所有项目合并后的全量原始行，计算最后的总 Median
    median_results = []  # 用于记录每个项目最终的汇总行

    print("\n📊 开始计算全局跨阈值的总中位数...")
    # 2. 遍历每个项目，把该项目在 9 个小文件夹里的所有数据纵向堆叠（Stack），一起求中位数
    for project_name in sorted(project_data_dict.keys()):
        files_list.append(project_name)

        # 将该项目在所有 cutoff 文件夹下的数据纵向拼接
        # 比如 9 个文件夹 * 30 轮 = 270 行数据
        project_total_matrix = np.vstack(project_data_dict[project_name])
        all_global_rows.append(project_total_matrix)

        # 计算该项目在全球（全阈值合集）实验下的总中位数
        res_raw = np.median(project_total_matrix, axis=0)

        # 时间统计 (假设最后一列是时间)
        m_time = np.median(project_total_matrix[:, -1])
        a_time = np.mean(project_total_matrix[:, -1])
        s_time = np.std(project_total_matrix[:, -1])

        # 拼接 31 个指标 + 3 个时间统计 = 34 列
        res = np.hstack([res_raw[:31], m_time, a_time, s_time])
        median_results.append(res)

    # 3. 计算最后一行大合集的 Median (所有小文件夹 × 所有项目 × 所有轮数的终极总中位数)
    global_large_matrix = np.vstack(all_global_rows)

    # 保存全量的总原始大矩阵 (包含了所有项目、所有阈值的混合行数据)
    results_df = DataFrame(global_large_matrix)
    results_df.to_csv(os.path.join(path, 'all_result_ONE_Global_Total.csv'), index=None, header=None)

    # 终极总中位数与时间统计
    median_all_raw = np.median(global_large_matrix, axis=0)
    m_time_all = np.median(global_large_matrix[:, -1])
    a_time_all = np.mean(global_large_matrix[:, -1])
    std_time_all = np.std(global_large_matrix[:, -1])

    median_all = np.hstack([median_all_raw[:31], m_time_all, a_time_all, std_time_all])
    median_results.append(median_all)

    # 4. 生成带有 'Median' 行的最终汇总 DataFrame
    data = DataFrame(median_results)
    files_list.append('Median')
    data.index = files_list

    # 定义 34 列标签
    measurename = [
        'precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC', 'accuracy',
        'Popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cMCC', 'cPMI', 'cIFA', 'cPCI', 'c_ROI_PII', 'c_ROI_PCI',
        'ceIFA',
        'mRecall', 'mPrecision', 'mfmeasure', 'mMCC', 'mPMI', 'mIFA', 'mPCI', 'm_ROI_PII', 'm_ROI_PCI', 'meIFA',
        'median_time', 'mean_time', 'std_time'
    ]
    data.columns = measurename

    # 直接保存在最外层的 AEEEM2020 根目录下
    output_file_path = os.path.join(path, 'result_ONE_Global_Total.csv')
    data.to_csv(output_file_path)

    print(f"\n✨ [全部完成] 全局总合集汇总成功！")
    print(f"📍 最终总中位数结果已生成至: {output_file_path}")