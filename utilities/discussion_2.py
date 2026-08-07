import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 设定输入与输出的数据文件夹路径
data_dir = r"E:\ICUSDP\INTC\ICUSDP\result_20260630_Discussion\replace2"

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
metrics_config = {
    'AUC': 4,
    'MCC': 8,
    'F-measure@20%LOC': 13,
    'IFA': 16
}

# 学术化图例名称映射字典
clusterer_name_map = {
    'KMEANS': 'K-means',
    'GMM': 'GMM',
    'KMEDOIDS': 'K-medoids',
    'SC': 'SC',
    'AGGLOMERATIVE': 'Agglomerative'
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
            raw_name = file_name.split("_")[-2].replace("-", "").upper()
            clusterer_name = clusterer_name_map.get(raw_name, raw_name)
        except Exception:
            clusterer_name = "Unknown"

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
                        'Clustering_Method': clusterer_name,
                        'Metric_Name': metric_name,
                        'Median_Score': median_value
                    })

    summary_df = pd.DataFrame(project_median_records)

    # 6. 配置美化画布与大字号控制
    plt.style.use('seaborn-v0_8-whitegrid')

    # 💡 画布高度设为 7.0，比例更协调
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    # ==================== 极大字号全局优化配置 ====================
    LABEL_FONT_SIZE = 22
    TICKS_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 15       # 微调横排图例字号，防止过于拥挤
    LEGEND_TITLE_SIZE = 16
    # =======================================================

    clusterer_order = ['K-means', 'GMM', 'K-medoids', 'SC', 'Agglomerative']
    metric_order = ['AUC', 'MCC', 'F-measure@20%LOC', 'IFA']

    # 7. 绘制簇状分组箱线图
    sns.boxplot(
        x='Metric_Name',
        y='Median_Score',
        hue='Clustering_Method',
        order=metric_order,
        hue_order=clusterer_order,
        data=summary_df,
        palette='Set2',
        width=0.6,
        fliersize=3.5,
        linewidth=1.5,
        ax=ax
    )

    # 8. 图表细节美化
    ax.set_xlabel('', labelpad=0)

    # 纵轴标签采用加粗大字体
    ax.set_ylabel('Median Values', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=12)

    # 极大化应用横纵轴刻度字号，并加粗刻度以提高印刷审阅清晰度
    ax.tick_params(axis='x', labelsize=TICKS_FONT_SIZE, labelfontfamily='sans-serif')
    ax.tick_params(axis='y', labelsize=TICKS_FONT_SIZE)

    # 针对横轴指标名称特别加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    ax.set_ylim(bottom=0.0)

    # 💡 核心修改：将图例横向平铺放置于图表正上方外部 (bbox_to_anchor 锚点调整)
    ax.legend(
        title='Clustering Method',
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),      # 精确定位在主图框顶部上方 2% 的位置
        ncol=5,                          # 5 个聚类方法横向一字排开
        frameon=True,
        columnspacing=0.8,               # 列间距控制
        borderpad=0.3
    )

    plt.tight_layout()

    # 9. 结果同步输出
    save_jpg_path = os.path.join(data_dir, "generalization_clustering_overall.jpg")
    save_pdf_path = os.path.join(data_dir, "generalization_clustering_overall.pdf")

    plt.savefig(save_jpg_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_pdf_path, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print(f"🎉 成功生成【顶部平铺图例 + 彻底零遮挡】的学术箱线图！")
    print(f"🖼️ 高清图片路径: {save_jpg_path}")
    print(f"📄 矢量 PDF 路径: {save_pdf_path}")
    print("=" * 60)