import os
import sys
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

if __name__ == '__main__':
    # --------------------------
    # 配置参数（AEEEM专属，无需修改）
    # --------------------------
    path = r"../result_20260124/unsupervised/"  # 与AEEEM CLA结果路径一致
    model_names = ['CLA']  # 仅处理CLA模型

    # --------------------------
    # 关键修改：在mean_time前添加median_time列
    # 15个核心指标 + median_time（1列） + mean_time（1列） + std_time（1列） = 18列
    # --------------------------
    # 1. 核心指标列名（15列，与AEEEM CLA输出一致）
    core_cols = [
        'precision', 'recall', 'pf', 'F1', 'AUC',
        'g_measure', 'g_mean', 'bal', 'MCC', 'Popt',
        'Erecall', 'Eprecision', 'Efmeasure', 'PMI', 'IFA'
    ]
    # 2. 最终输出列名（18列，含time统计值：median_time + mean_time + std_time）
    measurename_final = core_cols + ['median_time', 'mean_time', 'std_time']

    overall_results = []

    # 遍历每个模型（仅CLA）
    for model_name in model_names:
        model_dir = os.path.join(path, model_name)
        if not os.path.exists(model_dir):
            print(f"⚠️ 警告：{model_name}文件夹不存在，跳过！")
            continue

        # 筛选AEEEM的5个标准项目文件
        aeeem_projects = ['equinox', 'jdt', 'lucene', 'mylyn', 'pde']
        files = []
        for f in sorted(os.listdir(model_dir)):
            if f.endswith('.csv') and any(proj in f.lower() for proj in aeeem_projects):
                files.append(f)

        if not files:
            print(f"⚠️ 警告：{model_name}文件夹下无AEEEM项目文件，跳过！")
            continue

        files_num = len(files)
        print(f"\n📊 处理 {model_name} 方法（共{files_num}个AEEEM项目文件）")

        files_list = []
        project_median_list = []  # 存储每个项目的18列中位数结果

        # 处理每个AEEEM项目文件
        for file in files:
            file_path = os.path.join(model_dir, file)
            file_name = file[:-4]
            files_list.append(file_name)
            print(f"  处理项目: {file_name}")

            try:
                # --------------------------
                # 1. 读取带表头的AEEEM结果文件
                # --------------------------
                df = pd.read_csv(file_path, header=0, encoding='utf-8-sig')

                # 检查必要列是否存在
                required_cols = core_cols + ['time']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"    ⚠️ 缺失必要列: {missing_cols}，跳过该项目")
                    continue

                # --------------------------
                # 2. 数据清洗（转换数值类型，删除空值）
                # --------------------------
                # 核心指标列清洗
                core_data = df[core_cols].copy()
                core_data = core_data.apply(pd.to_numeric, errors='coerce').dropna()
                # time列清洗
                time_data = pd.to_numeric(df['time'], errors='coerce').dropna()

                # 检查有效数据量
                if len(core_data) == 0 or len(time_data) == 0:
                    print(f"    ⚠️ 无有效数据（核心指标/时间），跳过该项目")
                    continue

                # --------------------------
                # 3. 计算该项目的18列结果（新增median_time计算）
                # --------------------------
                # 3.1 核心指标的中位数（15列）
                core_median = core_data.median(axis=0).values  # 15个值
                # 3.2 time列的统计值（3列：中位数、均值、标准差）
                median_time = time_data.median()  # 新增：计算time列的中位数
                mean_time = time_data.mean()
                std_time = time_data.std()
                # 3.3 组合成18列数据（15 + 3）
                project_median = np.hstack([core_median, median_time, mean_time, std_time])  # 18个值
                project_median_list.append(project_median)

                print(f"    ✅ 处理成功（有效数据行数: {min(len(core_data), len(time_data))}）")

            except Exception as e:
                error_msg = str(e)[:80]
                print(f"    ❌ 处理失败: {error_msg}，跳过该项目")
                continue

        # 检查是否有有效项目数据
        if not project_median_list:
            print(f"⚠️ 警告：{model_name}无有效AEEEM项目数据，跳过！")
            continue

        # --------------------------
        # 4. 生成all_result文件（原始数据汇总）
        # --------------------------
        # 收集所有项目的原始核心指标+time数据
        all_raw_data = []
        for file in files:
            file_path = os.path.join(model_dir, file)
            try:
                df = pd.read_csv(file_path, header=0, encoding='utf-8-sig')
                core_data = df[core_cols].apply(pd.to_numeric, errors='coerce').dropna()
                time_data = pd.to_numeric(df['time'], errors='coerce').dropna()
                # 确保核心指标和time数据行数一致
                min_len = min(len(core_data), len(time_data))
                if min_len > 0:
                    raw = np.hstack([core_data.iloc[:min_len].values, time_data.iloc[:min_len].values.reshape(-1, 1)])
                    all_raw_data.append(raw)
            except:
                continue

        if all_raw_data:
            all_raw_arr = np.vstack(all_raw_data)
            all_raw_df = DataFrame(all_raw_arr, columns=core_cols + ['time'])
            all_raw_df.to_csv(os.path.join(path, f'all_result_{model_name}.csv'), index=None,
                              encoding='utf-8-sig')
            print(f"\n✅ 已保存AEEEM原始数据文件: all_result_{model_name}.csv")

        # --------------------------
        # 5. 生成项目中位数结果文件（含Median行）
        # --------------------------
        # 计算所有项目的汇总中位数（18列）
        median_all = np.median(np.array(project_median_list), axis=0)  # 18个值
        # 追加汇总中位数到列表（最后一行是Median）
        project_median_list.append(median_all)
        # 构建结果DataFrame
        result_df = DataFrame(project_median_list, columns=measurename_final)
        # 添加项目名索引（含Median行）
        files_list.append('Median')
        result_df.index = files_list

        # 保存中位数结果文件
        result_file = os.path.join(path, f'result_{model_name}_AEEEM.csv')
        result_df.to_csv(result_file, encoding='utf-8-sig')
        print(f"✅ 已保存AEEEM中位数结果文件: {os.path.basename(result_file)}")

        # --------------------------
        # 6. 记录整体中位数（用于模型汇总）
        # --------------------------
        overall_results.append(median_all)

    # --------------------------
    # 7. 生成模型汇总结果文件
    # --------------------------
    if overall_results:
        # 确保汇总数据列数与列名一致（18列）
        overall_arr = np.array(overall_results)
        if overall_arr.shape[1] == len(measurename_final):
            overall_df = DataFrame(overall_arr, columns=measurename_final)
            overall_df.index = model_names  # 模型名作为索引
            overall_file = os.path.join(path, 'result_allModels_AEEEM.csv')
            overall_df.to_csv(overall_file, encoding='utf-8-sig')
            print(f"\n🎉 所有AEEEM模型处理完成！汇总文件: {os.path.basename(overall_file)}")
        else:
            print(
                f"\n⚠️ 警告：汇总数据列数（{overall_arr.shape[1]}）与列名（{len(measurename_final)}）不匹配，未生成汇总文件")
    else:
        print(f"\n⚠️ 警告：无有效AEEEM模型数据，未生成汇总文件")