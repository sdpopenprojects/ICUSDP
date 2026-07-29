import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 设定输入与输出的数据文件夹路径（默认切回 replace1）
data_dir = r"F:\ICUSDP\INTC\ICUSDP\result_20260630_Discussion\replace1"
# data_dir = r"F:\ICUSDP\INTC\ICUSDP\result_20260630_Discussion\replace3_1"

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
# 💡 核心修改一：调整字典定义顺序，实现 MCC 与 IFA 位置互换
metrics_config = {
    'AUC': 4,
    'MCC': 8,              # 提前到第二位
    'IFA': 16,             # 延后到第三位
    'F-measure@20%LOC': 13
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
            learner_name = file_name.split("_")[-2].upper()
        except Exception:
            learner_name = "Unknown"

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
                        'Feature_Learner': learner_name,
                        'Metric_Name': metric_name,
                        'Median_Score': median_value
                    })

    summary_df = pd.DataFrame(project_median_records)

    # 6. 配置美化画布与大字号控制
    plt.style.use('seaborn-v0_8-whitegrid')

    # 💡 宽度设为 11.5，保证内嵌图例排版紧凑且风格一致
    fig, ax = plt.subplots(figsize=(11.5, 6.5))

    # ==================== 大字号同步放大配置 ====================
    LABEL_FONT_SIZE = 22
    TICKS_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 16
    LEGEND_TITLE_SIZE = 17
    # =======================================================

    learner_order = ['VAE', 'AE', 'CAE', 'DAE']
    # 💡 核心修改二：显式指定横轴指标排序，确保箱线图顺序为 AUC -> MCC -> IFA -> F-measure
    metric_order = ['AUC', 'MCC', 'IFA', 'F-measure@20%LOC']

    # 7. 绘制簇状分组箱线图
    sns.boxplot(
        x='Metric_Name',
        y='Median_Score',
        hue='Feature_Learner',
        order=metric_order,          # 🚀 应用横轴显式排序
        hue_order=learner_order,
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
    ax.set_ylabel('Median Values (IFA is log10-scaled)', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)

    # 横纵轴刻度字号放大
    ax.tick_params(axis='x', labelsize=TICKS_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_FONT_SIZE)

    # 横轴指标名称加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    ax.set_ylim(bottom=0.0)

    # 💡 核心修改三：移除下方外挂图例参数，改为内嵌在画布内部的右上方
    ax.legend(
        title='Feature Learner',
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc='upper right',                 # 完美定位在主图内部右上角
        frameon=True
    )

    plt.tight_layout()

    # 9. 结果同步输出到指定目录下
    save_jpg_path = os.path.join(data_dir, "generalization_feature_learner_overall_vae_first.jpg")
    save_pdf_path = os.path.join(data_dir, "generalization_feature_learner_overall_vae_first.pdf")

    plt.savefig(save_jpg_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_pdf_path, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print(f"🎉 成功更新特征学习器实验图表！【内嵌右上方图例 + MCC与IFA位置互换】")
    print(f"🖼️ 高清图片路径: {save_jpg_path}")
    print(f"📄 矢量 PDF 路径: {save_pdf_path}")
    print("=" * 60)