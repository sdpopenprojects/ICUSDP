import os
import glob
import pandas as pd
from scipy.stats import wilcoxon
import warnings

# 自动忽略 Wilcoxon 样本量较小时的内部警告，保持输出整洁
warnings.filterwarnings('ignore', message='Sample size too small')


def analyze_framework_combinations():
    # 1. 配置输入文件夹与输出结果文件夹路径
    input_dir = r"F:\ICUSDP\INTC\ICUSDP\total compare"
    output_dir = r"F:\ICUSDP\INTC\ICUSDP\result_main frame"

    # 如果输出文件夹不存在，则自动创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"【系统提示】已自动创建结果输出文件夹: {output_dir}")

    # 自动获取 input_dir 路径下所有的 Excel 文件 (.xlsx 或 .xls)
    excel_files = glob.glob(os.path.join(input_dir, "*.xlsx")) + glob.glob(os.path.join(input_dir, "*.xls"))

    if not excel_files:
        print(f"【错误提示】在路径 '{input_dir}' 下没有找到任何 Excel 文件，请检查路径是否正确。")
        return

    print(f"【系统提示】成功找到 {len(excel_files)} 个指标表格文件，正在为您执行非参数统计显著性检验...")

    # 定义我们的 4 种衍生组合和 1 种基准模型
    combinations = ['VAEsc', 'VAEkmeans', '双向sc', '双向kmeans']
    baseline = 'SC'

    # 设立一个全局容器，用于跨全部 15 个指标进行宏观投票决策
    all_metrics_summary = []

    # 2. 核心循环：依次处理每一个指标文件
    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        metric_name = os.path.splitext(file_name)[0]  # 以文件名作为指标名（例如 AUC, MCC 等）

        try:
            # 读取当前指标的 Excel 数据
            df = pd.read_excel(file_path)

            # 健壮性检查：确保该文件中包含了我们需要对比的所有列
            required_cols = [baseline] + combinations
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"【跳过文件】{file_name} 中缺少必要的列 {missing_cols}，已跳过。")
                continue

            file_results = []

            # 对 4 种组合分别和 SC 基准进行配对检验
            for combo in combinations:
                # 剔除含有空值的行（确保配对项目的数量严格一致）
                valid_data = df[[baseline, combo]].dropna()

                if len(valid_data) < 5:
                    print(f"【样本不足】文件 {file_name} 中的 {combo} 有效配对样本量太少，无法进行统计检验。")
                    continue

                # 执行双尾配对 Wilcoxon 符号秩检验
                stat, p_value = wilcoxon(valid_data[combo], valid_data[baseline], alternative='two-sided')

                # 统计当前指标在 28 个项目上有多少个项目性能优于、等于或劣于纯 SC
                win = (valid_data[combo] > valid_data[baseline]).sum()
                loss = (valid_data[combo] < valid_data[baseline]).sum()
                tie = (valid_data[combo] == valid_data[baseline]).sum()

                # 计算均值差异 (组合均值 - 基准均值) 观察宏观趋势
                mean_diff = valid_data[combo].mean() - valid_data[baseline].mean()

                # 显著性判定 (学术界通用 alpha = 0.05)
                is_significant = 'Yes' if p_value < 0.05 else 'No'

                # 记录该指标下该组合的详细统计特征
                file_results.append({
                    '组合名称': combo,
                    '均值差值(Combo-SC)': round(mean_diff, 4),
                    '胜/平/负(W/T/L)': f"{win}/{tie}/{loss}",
                    'P值(P-Value)': round(p_value, 4),
                    '是否显著差异(p<0.05)': is_significant
                })

                # 同步记录到宏观大表中，方便最后的跨指标投票
                all_metrics_summary.append({
                    '指标名称': metric_name,
                    '组合名称': combo,
                    '是否显著差异(p<0.05)': is_significant,
                    '是否整体占优(Mean_Diff>0)': 'Yes' if mean_diff > 0 else 'No',
                    '胜场数(Win)': win
                })

            # 将当前单项指标的 4 组详细对比数据导出为独立 Excel 报表
            detail_df = pd.DataFrame(file_results)
            output_file_path = os.path.join(output_dir, f"{metric_name}_vs_SC_显著性检验结果.xlsx")
            detail_df.to_excel(output_file_path, index=False)
            print(f" -> 成功导出单项指标统计：{os.path.basename(output_file_path)}")

        except Exception as e:
            print(f"【处理异常】读取或计算文件 {file_name} 时发生错误: {str(e)}")

    # 3. 终极宏观决策：跨越全部 15 个指标进行总票数统计
    if all_metrics_summary:
        summary_df = pd.DataFrame(all_metrics_summary)

        decision_table = []
        for combo in combinations:
            combo_data = summary_df[summary_df['组合名称'] == combo]

            # 条件一：不仅 p < 0.05（显著），而且均值差值 > 0（代表显著变得更好，而不是显著变差）
            sig_wins = combo_data[(combo_data['是否显著差异(p<0.05)'] == 'Yes') & (
                        combo_data['是否整体占优(Mean_Diff>0)'] == 'Yes')].shape[0]
            # 条件二：均值超越 SC 的总指标数（含微弱优势、未达显著的指标）
            overall_better_metrics = combo_data[combo_data['是否整体占优(Mean_Diff>0)'] == 'Yes'].shape[0]
            # 条件三：在所有文件、所有项目上累计的总胜场（Win）次数
            total_wins_count = combo_data['胜场数(Win)'].sum()

            decision_table.append({
                '组合名称': combo,
                '统计学显著优于SC的指标数量 (关键项)': sig_wins,
                '均值超越SC的指标总体数量': overall_better_metrics,
                '所有项目累计总胜场数(Total Wins)': total_wins_count
            })

        # 按照“统计学显著优于SC的指标数量”和“总胜场数”从大到小排序
        decision_df = pd.DataFrame(decision_table).sort_values(
            by=['统计学显著优于SC的指标数量 (关键项)', '所有项目累计总胜场数(Total Wins)'],
            ascending=False
        )

        # 将最终的框架选型选拔推荐总表保存到结果目录
        decision_output_path = os.path.join(output_dir, "00_ICUSDP主框架模型选型决策推荐总表.xlsx")
        decision_df.to_excel(decision_output_path, index=False)

        print("\n" + "=" * 75)
        print("【ICUSDP 框架全指标非参数统计检验综合选型结果】")
        print("=" * 75)
        print(decision_df.to_string(index=False))
        print("=" * 75)
        print(f"选型决策总结表已保存至: {decision_output_path}")
        print("\n【研究结论与主实验模型选择建议】：")
        best_combo = decision_df.iloc[0]['组合名称']
        print(f" 根据上面综合多指标投票结果，推荐优先选择【 {best_combo} 】作为您论文或报告中 ICUSDP 的主框架组合！")
        print(
            " 因为该组合在统计学检验中，获得了最多指标的显著性优势(p<0.05)，且在具体项目上胜率最高，支撑主实验最具说服力。")


if __name__ == "__main__":
    analyze_framework_combinations()