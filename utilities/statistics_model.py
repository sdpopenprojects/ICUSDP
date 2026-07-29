import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings

# 忽略 Wilcoxon 样本量较小或零差值时的警告
warnings.filterwarnings('ignore', category=UserWarning)

# ==================== 路径与配置区域 ====================
# 输入文件夹：存放 34 或 41 个指标独立汇总表的文件夹 (此时应该是已经剔除 VAE 和 CAE 的 16 列模型表)
INPUT_DIR = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\all_over"
# 输出文件夹：统计结果与图片保存的文件夹
OUTPUT_DIR = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\statistic"

# 你的主模型列名
MAIN_MODEL = 'ICUSDP'


# =======================================================

def cliffs_delta(x, y):
    """计算 Cliff's Delta 效应量"""
    n_x = len(x)
    n_y = len(y)
    more = sum([1 for i in x for j in y if i > j])
    less = sum([1 for i in x for j in y if i < j])
    delta = (more - less) / (n_x * n_y)
    return delta


def effect_size_interpret(delta):
    """Cliff's Delta 效应量等级解读"""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return 'Negligible'
    elif abs_delta < 0.33:
        return 'Small'
    elif abs_delta < 0.474:
        return 'Medium'
    else:
        return 'Large'


def stat_compare(df, metric_name):
    """对单个指标的表格进行统计检验"""
    results = []

    if MAIN_MODEL not in df.columns:
        print(f"⚠️ 警告: 指标 [{metric_name}] 中未找到主模型 [{MAIN_MODEL}]，跳过统计检验。")
        return pd.DataFrame(results)

    for col in df.columns:
        if col != MAIN_MODEL and col.lower() != 'project':
            pair_data = df[[MAIN_MODEL, col]].dropna()

            if len(pair_data) < 10:
                continue

            x_vals = pair_data[MAIN_MODEL].values
            y_vals = pair_data[col].values

            if np.array_equal(x_vals, y_vals):
                p_value = 1.0
                delta = 0.0
            else:
                try:
                    _, p_value = wilcoxon(x_vals, y_vals)
                except Exception:
                    p_value = np.nan

                delta = cliffs_delta(x_vals, y_vals)

            es = effect_size_interpret(delta)

            results.append({
                'Metric': metric_name,
                'Compare_Model': col,
                f'Median_{MAIN_MODEL}': np.median(x_vals),
                f'Median_{col}': np.median(y_vals),
                'Wilcoxon_p_value': p_value,
                'Statistically_Significant': 'Yes (p<0.05)' if p_value < 0.05 else 'No',
                'Cliffs_Delta': delta,
                'Effect_Size_Level': es
            })

    return pd.DataFrame(results)


def plot_boxplot(df, metric_name, save_dir):
    """
    【升级：解决图例遮挡】
    改变图例位置至图形正上方，并水平一字排开，绝不遮挡任何数据箱线和异常值点
    """
    plot_df = df.copy()
    if 'Project' in plot_df.columns:
        plot_df = plot_df.drop(columns=['Project'])

    # 1. 定义四大类别的科研配色
    color_map = {
        'main': '#E64B35',  # 鲜艳红：突出你的主模型 ICUSDP
        'one': '#4DBBD5',  # 天蓝色：代表独树一帜的 ONE 基准
        'supervised': '#00A087',  # 翡翠绿：代表经典有监督机器学习方法
        'unsupervised': '#F39B7F'  # 浅桔色：代表经典无监督/半监督方法
    }

    # 2. 为当前表格现存的模型列构建颜色列表
    palette_colors = []
    for model in plot_df.columns:
        if model == 'ICUSDP':
            palette_colors.append(color_map['main'])
        elif model == 'ONE':
            palette_colors.append(color_map['one'])
        elif model in ['DT', 'GBM', 'linearSVM', 'LR', 'RF', 'XGBoost']:
            palette_colors.append(color_map['supervised'])
        elif model in ['CLA', 'CLAMI', 'KMedoids', 'MD', 'MU', 'SC', 'TCL', 'TCLP']:
            palette_colors.append(color_map['unsupervised'])
        else:
            palette_colors.append('#CCCCCC')

    # 3. 开始绘图
    fig, ax = plt.subplots(figsize=(13, 6.8))  # 稍微加高一点点点纵向空间给顶部图例

    # 传入定制颜色列表 palette_colors
    sns.boxplot(data=plot_df, palette=palette_colors, linewidth=1.2, width=0.6, ax=ax)

    # 稍微抬高标题，给下方的水平图例留出呼吸空间 (pad=35)
    plt.title(f'Performance Distribution across Models - {metric_name}', fontsize=14, fontweight='bold', pad=35)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.xlabel('Models', fontsize=12, fontweight='bold')

    # 添加网格背景
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 4. 【核心修复：将图例置于上方并水平排开】
    legend_patches = [
        mpatches.Patch(color=color_map['main'], label='ICUSDP'),
        mpatches.Patch(color=color_map['one'], label='ONE'),
        mpatches.Patch(color=color_map['supervised'], label='Supervised'),
        mpatches.Patch(color=color_map['unsupervised'], label='Unsupervised')
    ]

    # loc='lower center': 图例锚点在图例框的底部中心
    # bbox_to_anchor=(0.5, 1.02): 将图例框精密锁定在绘图区正上方 2% 的空白位置
    # ncol=4: 强制图例并排成一整行，不占用图表内部的任何高度
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=4, fontsize=10.5, frameon=False)

    plt.tight_layout()

    # 独立保存图片到 statistic 目录
    img_path = os.path.join(save_dir, f"{metric_name}_boxplot.png")
    plt.savefig(img_path, dpi=300)
    plt.close()  # 释放内存


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建统计输出文件夹: {OUTPUT_DIR}")

    excel_files = glob.glob(os.path.join(INPUT_DIR, "*.xlsx"))
    if not excel_files:
        print(f"❌ 错误: 在路径 {INPUT_DIR} 中没有找到任何 .xlsx 指标文件！")
        return

    all_metric_results = []
    total_files = len(excel_files)

    print(f"================== 开始处理，共检测到 {total_files} 个指标表 ==================")

    for idx, file_path in enumerate(excel_files, 1):
        file_name = os.path.basename(file_path)
        metric_name = os.path.splitext(file_name)[0]

        try:
            df = pd.read_excel(file_path)

            # 1. 运行统计对比
            df_stat = stat_compare(df, metric_name)
            if not df_stat.empty:
                all_metric_results.append(df_stat)

            # 2. 运行四色画图逻辑（水平图例不遮挡数据）并保存
            plot_boxplot(df, metric_name, OUTPUT_DIR)

            print(f"[{idx}/{total_files}] 成功完成指标 [{metric_name}] 的统计检验与无遮挡箱线图绘制。")

        except Exception as e:
            print(f"❌ 处理文件失败 {file_name}: {e}")

    # 合并所有指标的检验大表并输出
    if all_metric_results:
        df_final_report = pd.concat(all_metric_results, ignore_index=True)
        report_file_path = os.path.join(OUTPUT_DIR, "SDP_all_statistical_results.xlsx")
        df_final_report.to_excel(report_file_path, index=False)

        print("\n==============================================================")
        print(f"🎉 🎉 【无遮挡图例升级任务完美完成！】")
        print(f"📊 统计检验大表保存在: 📂 {report_file_path}")
        print(f"🖼️  {total_files} 个指标的【水平无遮挡图例科研箱线图】已全部输出至: 📂 {OUTPUT_DIR}")
        print("==============================================================")
    else:
        print("⚠️ 未能生成任何有效的统计分析结果，请检查输入的汇总表数据。")


if __name__ == "__main__":
    main()