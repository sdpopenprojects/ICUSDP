import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置学术论文出版级字体，确保排版美观、规整
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 指标名称规范化映射表 ====================
METRIC_NAME_MAP = {
    'precision': 'Precision',
    'recall': 'Recall',
    'pf': 'Probability of False Alarm (PF)',
    'F1': 'F1-Measure',
    'AUC': 'AUC',
    'g_measure': 'G-Measure',
    'g_mean': 'G-Mean',
    'bal': 'Balance(Bal)',
    'MCC': 'MCC',
    'accuracy': 'Accuracy',
    'Popt': '$P_{opt}$',
    'cEfmeasure': 'F-measure@20%LOC',
    'cIFA': 'IFA',
    'cMCC': 'MCC@20%LOC',
    'mMCC': 'MCC@20%Modules',
    'ceIFA': 'eIFA',
    'c_ROI_PII': 'ROI@20%LOC',
    'm_ROI_PCI': 'ROI@20%Modules',
    'time': 'Execution Time (s)',
}
# =============================================================

# ==================== 1. 定义指定的绝对路径 ====================
input_dir = r'E:\ICUSDP-main\ICUSDP-main\new2\Radar'
output_dir = r'E:\ICUSDP-main\ICUSDP-main\new2\Radar'

excel_path = os.path.join(input_dir, 'all.xlsx')
output_pdf_path = os.path.join(
    output_dir, 'Radar_Comparison.pdf'
)

# ==================== 2. 读取Excel工作表与数据清洗 ====================
df_raw = pd.read_excel(excel_path)
method_col_name = df_raw.columns[0]
df_raw = df_raw.set_index(method_col_name)

# 锁定指定的 4 个核心度量指标
metrics = ['AUC', 'cIFA', 'MCC', 'cEfmeasure']
df = df_raw[metrics].copy()

# cIFA 物理语义是越小越好，取负号转化为越大越好（便于雷达图统一向外扩张）
df['cIFA'] = -df['cIFA']

# 全局统一进行 Min-Max 归一化，映射至 [0.15, 0.95] 区间
df_norm = df.copy()
for col in df.columns:
  min_val = df[col].min()
  max_val = df[col].max()
  if max_val != min_val:
    df_norm[col] = 0.15 + (df[col] - min_val) / (max_val - min_val) * 0.80
  else:
    df_norm[col] = 0.9

# ==================== 3. 划分三大阵营并计算中位数 ====================
# 🔥 修改点：在无监督列表中同时保留 MUSDP 和 KMedoids
unsupervised_list = [
    'ONE',
    'CLA',
    'CLAMI',
    'MUSDP',
    'KMedoids',
    'MD',
    'MU',
    'SC',
    'TCL',
    'TCLP',
]
supervised_list = ['DT', 'GBM', 'linearSVM', 'LR', 'RF', 'XGBoost']

# 过滤数据集中实际存在的模型，防止因缺失某个模型名称报错
existing_unsupervised = [
    m for m in unsupervised_list if m in df_norm.index
]
existing_supervised = [m for m in supervised_list if m in df_norm.index]

unsupervised_profile = (
    df_norm.loc[existing_unsupervised].median(axis=0).values.tolist()
)
supervised_profile = (
    df_norm.loc[existing_supervised].median(axis=0).values.tolist()
)

if 'ICUSDP' in df_norm.index:
  icusdp_profile = df_norm.loc['ICUSDP'].values.tolist()
else:
  raise KeyError("在数据集中未找到 'ICUSDP' 模型，请检查第一列的拼写。")

# ==================== 4. 雷达图闭环几何配置 ====================
display_labels = [METRIC_NAME_MAP.get(m, m) for m in metrics]

num_vars = len(display_labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# 首尾闭环
angles += angles[:1]
unsupervised_profile += unsupervised_profile[:1]
supervised_profile += supervised_profile[:1]
icusdp_profile += icusdp_profile[:1]

# ==================== 5. 开始绘制雷达图 ====================
fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

# --------- 5.1 绘制无监督阵营中位数线 (科技蓝，虚线) ---------
ax.plot(
    angles,
    unsupervised_profile,
    color='#1976D2',
    linewidth=2.2,
    linestyle='--',
    marker='s',
    markersize=5,
    label='Unsupervised Baselines (Median)',
)
ax.fill(angles, unsupervised_profile, color='#1976D2', alpha=0.05)

# --------- 5.2 绘制监督阵营中位数线 (学术绿，点划线) ---------
ax.plot(
    angles,
    supervised_profile,
    color='#388E3C',
    linewidth=2.2,
    linestyle='-.',
    marker='^',
    markersize=5,
    label='Supervised Baselines (Median)',
)
ax.fill(angles, supervised_profile, color='#388E3C', alpha=0.04)

# --------- 5.3 绘制 ICUSDP 核心主模型 (高亮学术红，破圈粗实线) ---------
ax.plot(
    angles,
    icusdp_profile,
    color='#D32F2F',
    linewidth=4.0,
    linestyle='-',
    marker='o',
    markersize=8,
    zorder=10,
    label='ICUSDP (Ours)',
)
ax.fill(angles, icusdp_profile, color='#D32F2F', alpha=0.18, zorder=10)

# ==================== 6. 坐标轴及排版大字号美化 ====================
ax.set_xticks(angles[:-1])

RADAR_LABEL_SIZE = 16  # 雷达图四周指标标签字号
RADAR_LEGEND_SIZE = 13  # 下方图例字体字号

# 文字与外圈间距保持 35
ax.tick_params(axis='x', which='major', pad=35)

# 定制 4 个方向标签的水平/垂直对齐方案
for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
  angle_deg = np.rad2deg(angle)
  if angle_deg == 0:
    label.set_horizontalalignment('left')
  elif angle_deg == 180:
    label.set_horizontalalignment('right')
  elif 0 < angle_deg < 180:
    label.set_horizontalalignment('center')
    label.set_verticalalignment('bottom')
  else:
    label.set_horizontalalignment('center')
    label.set_verticalalignment('top')

# 应用大字号标签
ax.set_xticklabels(
    display_labels, fontsize=RADAR_LABEL_SIZE, fontweight='bold', color='#262626'
)

# 隐藏内圈数字标签，设置极坐标轴界限
ax.set_yticklabels([])
ax.set_rlim(0, 1.05)

# 美化背景蜘蛛网格线
ax.grid(True, color='#D9D9D9', linestyle='--', linewidth=0.8)
ax.spines['polar'].set_color('#AEAEAE')

# 放大图例字号，并微调 Y 轴偏移 (-0.14) 匹配大字体排版
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.14),
    ncol=3,
    fontsize=RADAR_LEGEND_SIZE,
    frameon=True,
    edgecolor='#E0E0E0',
    facecolor='#FAFAFA',
)

# 手动压缩子图的底部边缘，给放大的图例留出更充裕的画布空间
plt.subplots_adjust(bottom=0.22, top=0.90, left=0.10, right=0.90)

# ==================== 7. 确保输出目录存在并导出 ====================
os.makedirs(output_dir, exist_ok=True)

plt.savefig(output_pdf_path, dpi=300, bbox_inches='tight')
print(f'🎉 雷达图绘制成功！已导出至: {output_pdf_path}')
plt.show()