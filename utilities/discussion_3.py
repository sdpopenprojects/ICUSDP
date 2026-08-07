import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 设定输入与输出的数据文件夹路径
data_dir = r"E:\ICUSDP\INTC\ICUSDP\result_20260630_Discussion\replace3_1"

# 2. 定义 28 个软件项目的标准追加写入顺序
project_list = [
    'activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
    'camel-1.4.0', 'camel-1.6.0', 'derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1',
    'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3', 'poi-1.5', 'poi-2.0', 'poi-2.5', 'poi-3.0',
    'prop-1', 'prop-2', 'prop-3', 'prop-4', 'prop-5', 'tomcat', 'velocity-1.4', 'velocity-1.5',
    'velocity-1.6', 'wicket-1.3.0'
]
reps = 100  # 每轮重复次数

# 3. 核心指标列索引映射
# 💡 核心修改一：调整字典顺序，F-measure@20%LOC 在前，IFA 延后到最后
metrics_config = {
    'AUC': 4,
    'MCC': 8,
    'F-measure@20%LOC': 13,
    'IFA': 16
}

project_median_records = []

# 4. 扫描并批量读取文件
search_pattern = os.path.join(data_dir, "all_result_*.csv")
file_paths = glob.glob(search_pattern)

if not file_paths:
    print(f"❌ Error: No files matching all_result_*.csv found in {data_dir}!")
else:
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            interpreter_name = file_name.split("_")[-2].upper()
        except Exception:
            interpreter_name = "Unknown"

        df_raw = pd.read_csv(file_path, header=None)

        for i, project_name in enumerate(project_list):
            start_idx = i * reps
            end_idx = start_idx + reps

            if start_idx < len(df_raw):
                df_chunk = df_raw.iloc[start_idx:end_idx]

                for metric_name, col_idx in metrics_config.items():
                    median_value = df_chunk.iloc[:, col_idx].median()

                    if metric_name == 'IFA':
                        median_value = np.log10(median_value + 1)

                    project_median_records.append({
                        'Project': project_name,
                        'Interpreter_Method': interpreter_name,
                        'Metric_Name': metric_name,
                        'Median_Score': median_value
                    })

    summary_df = pd.DataFrame(project_median_records)

    # 6. 配置美化画布与大字号控制
    plt.style.use('seaborn-v0_8-whitegrid')

    # 💡 画布高度设为 7.0，与组件一、二保持一致
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    # ==================== 大字号同步放大配置 ====================
    LABEL_FONT_SIZE = 22
    TICKS_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 15       # 微调字号适配横向图例
    LEGEND_TITLE_SIZE = 16
    # =======================================================

    interpreter_order = ['DT', 'RF', 'GBM', 'XGBOOST']
    # 💡 核心修改二：指定横轴指标排序为 AUC -> MCC -> F-measure@20%LOC -> IFA
    metric_order = ['AUC', 'MCC', 'F-measure@20%LOC', 'IFA']

    # 7. 绘制簇状分组箱线图
    sns.boxplot(
        x='Metric_Name',
        y='Median_Score',
        hue='Interpreter_Method',
        order=metric_order,          # 🚀 应用全新横轴排序逻辑
        hue_order=interpreter_order,
        data=summary_df,
        palette='Set2',
        width=0.5,
        fliersize=3.5,
        linewidth=1.2,
        ax=ax
    )

    # 8. 图表细节美化
    ax.set_xlabel('', labelpad=0)

    # 纵轴标签字号保持放大
    ax.set_ylabel('Median Values', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # 横纵轴刻度字号放大
    ax.tick_params(axis='x', labelsize=TICKS_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_FONT_SIZE)

    # 针对横轴指标名称特别加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    ax.set_ylim(bottom=0.0)

    # 💡 核心修改三：将图例横向一字平铺置于主图正上方（ncol=4）
    ax.legend(
        title='Interpreter',
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),      # 精确定位在主图顶部上方 2% 位置
        ncol=4,                          # 4 个解释器模型横向一字排开
        frameon=True,
        columnspacing=1.0,               # 列间距控制
        borderpad=0.3
    )

    plt.tight_layout()

    # 9. 结果同步输出（含防文件占用捕获）
    save_jpg_path = os.path.join(data_dir, "generalization_classifier_overall.jpg")
    save_pdf_path = os.path.join(data_dir, "generalization_classifier_overall.pdf")

    # 保存 JPG
    try:
        plt.savefig(save_jpg_path, dpi=300, bbox_inches='tight')
        print(f"🖼️ 高清图片保存成功: {save_jpg_path}")
    except PermissionError:
        print(f"⚠️ JPG 文件被占用，请关闭相关预览软件: {save_jpg_path}")

    # 保存 PDF
    try:
        plt.savefig(save_pdf_path, bbox_inches='tight')
        print(f"📄 矢量 PDF 保存成功: {save_pdf_path}")
    except PermissionError:
        alt_pdf_path = os.path.join(data_dir, "generalization_classifier_overall_new.pdf")
        plt.savefig(alt_pdf_path, bbox_inches='tight')
        print(f"⚠️ 原 PDF 文件被占用！已临时保存至新路径: {alt_pdf_path}")

    plt.close()

    print("\n" + "=" * 60)
    print(f"🎉 成功同步更新组件三图表！【顶部平铺图例 + IFA与F-measure位置互换】")
    print("=" * 60)