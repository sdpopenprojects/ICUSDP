import os
import sys
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

if __name__ == '__main__':
    # --------------------------
    # 配置参数（根据实际路径修改）
    # --------------------------
    # path = r"../result_20260115/unsupervised/"  # 结果根路径
    path = r"../result_20260124/unsupervised/"  # 结果根路径
    # 模型列表（包含CLA和其他方法）
    model_names = ['CLA']

    # --------------------------
    # 核心修复：更新列名列表（匹配19列）
    # 原18列 + randseed（随机种子列）= 19列
    # --------------------------
    measurename = [
        'precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC',
        'popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA',
        'median_time', 'mean_time', 'std_time', 'randseed'  # 新增：randseed列
    ]

    # 兼容处理：如果某些方法没有randseed列，使用18列的列名
    measurename_18 = [
        'precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC',
        'popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cPMI', 'cIFA',
        'median_time', 'mean_time', 'std_time'
    ]

    overall_results = []

    # 遍历每个模型
    for model_name in model_names:
        model_dir = os.path.join(path, model_name)
        if not os.path.exists(model_dir):
            print(f"⚠️ 警告：{model_name}文件夹不存在，跳过！")
            continue

        # 获取该模型下的所有CSV文件
        files = [f for f in sorted(os.listdir(model_dir)) if f.endswith('.csv')]
        if not files:
            print(f"⚠️ 警告：{model_name}文件夹下无CSV文件，跳过！")
            continue

        files_num = len(files)
        print(f"\n📊 处理 {model_name} 方法（共{files_num}个项目文件）")

        files_list = []
        results = []
        median_results = []

        # 处理每个项目文件
        for file in files:
            file_path = os.path.join(model_dir, file)
            file_name = file[:-4]
            files_list.append(file_name)
            print(f"  处理项目: {file_name}")

            try:
                # 读取CSV文件（无表头）
                df = pd.read_csv(file_path, header=None)

                # --------------------------
                # 关键修复1：动态适配列数
                # --------------------------
                df_cols = df.shape[1]
                if df_cols == 16:
                    # 原始格式：15指标 + 1时间（无统计列）
                    # 转换为数值类型，处理可能的空值
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    results.append(df.values)

                    # 计算中位数（指标列）
                    res = np.median(df.values, axis=0)

                    # 计算时间统计（最后一列）
                    mean_time = np.mean(df.iloc[:, -1], axis=0)
                    std_time = np.std(df.iloc[:, -1], axis=0)
                    res = np.hstack([res, mean_time, std_time])

                    # 补充randseed列（默认0）
                    res = np.hstack([res, 0])

                elif df_cols == 17:
                    # CLA新增格式：16列（含randseed） + 无统计列
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    results.append(df.values)

                    # 计算中位数（前15指标列）
                    res = np.median(df.iloc[:, :-2].values, axis=0)  # 排除time和randseed

                    # 时间统计（倒数第二列）
                    mean_time = np.mean(df.iloc[:, -2], axis=0)
                    std_time = np.std(df.iloc[:, -2], axis=0)
                    res = np.hstack([res, mean_time, std_time])

                    # randseed列（最后一列）取均值
                    randseed_mean = np.mean(df.iloc[:, -1], axis=0)
                    res = np.hstack([res, randseed_mean])

                elif df_cols == 19:
                    # 已有统计列的格式
                    df = df.apply(pd.to_numeric, errors='coerce').dropna()
                    results.append(df.values)
                    res = np.median(df.values, axis=0)

                else:
                    print(f"    ⚠️ 列数不支持（{df_cols}列），跳过该项目")
                    continue

                median_results.append(res)

            except Exception as e:
                print(f"    ❌ 处理失败: {str(e)[:50]}，跳过该项目")
                continue

        # 检查是否有有效数据
        if not results:
            print(f"⚠️ 警告：{model_name}无有效数据，跳过！")
            continue

        # 保存all_result文件
        results_arr = np.vstack(results)
        results_df = DataFrame(results_arr)
        results_df.to_csv(os.path.join(path, f'all_result_{model_name}.csv'), index=None, header=None)

        # 计算整体中位数
        median_all = np.median(results_df, axis=0)

        # 补充到中位数列表
        median_results.append(median_all)
        data = DataFrame(median_results)

        # 添加Median行
        files_list.append('Median')
        data.index = files_list

        # --------------------------
        # 关键修复2：根据实际列数分配列名
        # --------------------------
        data_cols = data.shape[1]
        if data_cols == 18:
            data.columns = measurename_18
        elif data_cols == 19:
            data.columns = measurename
        else:
            # 截断或补充列名
            if data_cols > 19:
                data = data.iloc[:, :19]
                data.columns = measurename
            else:
                # 补充缺失列名
                add_cols = ['unknown_col_' + str(i) for i in range(19 - data_cols)]
                data.columns = measurename[:data_cols] + add_cols

        # 保存该模型的结果文件
        data.to_csv(os.path.join(path, f'result_{model_name}.csv'), encoding='utf-8-sig')
        print(f"✅ 已保存 {model_name} 结果文件: result_{model_name}.csv")

        # 记录整体中位数（用于汇总）
        overall_results.append(median_all)

    # 生成所有模型的汇总结果
    if overall_results:
        overall_df = DataFrame(overall_results)
        overall_df.index = [name for name in model_names if os.path.exists(os.path.join(path, name))]

        # 适配汇总表列名
        if overall_df.shape[1] == 18:
            overall_df.columns = measurename_18
        else:
            overall_df.columns = measurename[:overall_df.shape[1]]

        overall_df.to_csv(os.path.join(path, 'result_allModels.csv'), encoding='utf-8-sig')
        print(f"\n🎉 所有模型处理完成！汇总文件已保存: result_allModels.csv")
    else:
        print(f"\n⚠️ 无有效模型数据，未生成汇总文件")