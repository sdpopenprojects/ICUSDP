import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------- 1. 基础设置与科研配色 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# 路径锁定
INPUT_FOLDER = Path(r"F:\ICUSDP\INTC\ICUSDP\W_C_test\visual\visual_data")
OUTPUT_FOLDER = Path(r"F:\ICUSDP\INTC\ICUSDP\W_C_test\visual\visual_result")
BAR_FOLDER = OUTPUT_FOLDER / "分组指标条形图（中位数）"
[folder.mkdir(exist_ok=True, parents=True) for folder in [OUTPUT_FOLDER, BAR_FOLDER]]

# 16 个模型的标准分类
METHOD_CLASSIFICATION = {
    "监督学习": {
        "ONE": "all_result_ONE.csv",
        "DT": "all_result_DT.csv", "GBM": "all_result_GBM.csv",
        "linearSVM": "all_result_linearSVM.csv", "LR": "all_result_LR.csv",
        "RF": "all_result_RF.csv", "XGBoost": "all_result_XGBoost.csv"
    },
    "无监督学习": {
        "CLA": "all_result_CLA.csv", "CLAMI": "all_result_CLAMI.csv",
        "KMedoids": "all_result_KMedoids.csv", "MD": "all_result_MD.csv",
        "MU": "all_result_MU.csv", "SC": "all_result_SC.csv",
        "TCL": "all_result_TCL.csv", "TCLP": "all_result_TCLP.csv"
    },
    "本文方法(ICUSDP)": {
        "ICUSDP": "all_result_ICUSDP.csv"
    }
}

COLOR_MAP = {
    "监督学习": "#3498db", "无监督学习": "#2ecc71", "本文方法(ICUSDP)": "#e74c3c"
}

# 精确还原你的 31 个原始指标名称
REAL_METRICS = [
    "precision", "recall", "pf", "F1", "AUC",
    "g_measure", "g_mean", "bal", "MCC", "accuracy",
    "Popt", "cErecall", "cEprecision", "cEfmeasure", "cMCC",
    "cPMI", "cIFA", "cPCI", "c_ROI_PII", "c_ROI_PCI",
    "ceIFA", "mRecall", "mPrecision", "mfmeasure", "mMCC",
    "mPMI", "mIFA", "mPCI", "m_ROI_PII", "m_ROI_PCI",
    "meIFA"
]


# -------------------------- 2. 读取数据与矩阵构建 --------------------------
def load_grouped_data():
    all_methods = []
    all_data_list = []
    method_to_group = {}

    for group_name, method_dict in METHOD_CLASSIFICATION.items():
        for method_name, filename in method_dict.items():
            file_path = INPUT_FOLDER / filename
            if file_path.exists():
                df = pd.read_csv(file_path, header=None).iloc[:, :len(REAL_METRICS)]
                df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
                all_methods.append(method_name)
                all_data_list.append(df.values)
                method_to_group[method_name] = group_name
                print(f"✅ 成功读取模型：{method_name}")
            else:
                print(f"⚠️ 找不到文件: {filename}，请检查该模型是否存在。")

    return all_methods, all_data_list, method_to_group


all_methods, all_data_list, method_to_group = load_grouped_data()

median_matrix = np.array([np.median(data, axis=0) for data in all_data_list])
iqr_matrix = np.array([np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0) for data in all_data_list])


# -------------------------- 3. 条形图模块（保持不变） --------------------------
def plot_single_grouped_bar():
    print("📊 开始生成 31 个原始指标独立条形图...")
    group_colors = [COLOR_MAP[method_to_group[m]] for m in all_methods]

    for metric_idx, metric_name in enumerate(REAL_METRICS):
        metric_medians = median_matrix[:, metric_idx]
        metric_iqrs = iqr_matrix[:, metric_idx]
        x = np.arange(len(all_methods))

        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(x, metric_medians, 0.6, yerr=metric_iqrs / 2, capsize=4, color=group_colors, alpha=0.85,
                      edgecolor='black', linewidth=0.5)

        ax.grid(axis='y', linestyle='--', alpha=0.5)
        legend_patches = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MAP[g], label=g) for g in COLOR_MAP]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=10, frameon=True)

        group_sizes = [len([m for m in all_methods if method_to_group[m] == g]) for g in METHOD_CLASSIFICATION]
        sep_positions = [sum(group_sizes[:i]) - 0.5 for i in range(1, len(group_sizes)) if group_sizes[i - 1] > 0]
        for pos in sep_positions:
            ax.axvline(x=pos, color='black', linestyle='--', linewidth=1.2, alpha=0.6)

        ax.set_title(f"Performance Comparison - {metric_name}", fontsize=14, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=12, fontweight="bold")

        max_val = np.max(metric_medians)
        if max_val > 1.5:
            ax.set_ylim(0, max_val * 1.25)
            for bar, val in zip(bars, metric_medians):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max_val * 0.01), f"{val:.1f}",
                        ha='center', fontsize=9)
        else:
            ax.set_ylim(0, max(1.1, max_val * 1.15))
            for bar, val in zip(bars, metric_medians):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha='center',
                        fontsize=9)

        plt.tight_layout()
        plt.savefig(BAR_FOLDER / f"{metric_name}_条形图.png", dpi=300)
        plt.close()
    print(f"✅ 条形图已成功保存至：{BAR_FOLDER}\n")


# -------------------------- 4. 雷达图模块（🌟 完美修复尺寸与标题显示问题） --------------------------
def plot_radar_chart():
    print("🕸️ 开始生成核心多维综合雷达图...")
    radar_labels = ["precision", "recall", "F1", "AUC", "g_mean"]
    metric_indices = [REAL_METRICS.index(name) for name in radar_labels]

    group_profiles = {}
    for g_name in METHOD_CLASSIFICATION.keys():
        indices = [i for i, m in enumerate(all_methods) if method_to_group[m] == g_name]
        if indices:
            group_profiles[g_name] = np.mean(median_matrix[indices][:, metric_indices], axis=0)

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    angles += angles[:1]

    # 🌟 微调画布尺寸为 8.5 x 8.5 保持黄金比例
    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))

    for g_name, profile in group_profiles.items():
        values = profile.tolist()
        values += values[:1]
        ax.plot(angles, values, color=COLOR_MAP[g_name], linewidth=2.5, label=g_name, marker='o', markersize=6)
        ax.fill(angles, values, color=COLOR_MAP[g_name], alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)

    # 🌟 修复关键点 1：调整 pad 与 y 轴相对坐标，给标题预留空间防止被顶部挡住
    plt.title("Overall Performance Radar Profiling", fontsize=14, fontweight="bold", pad=35, y=1.02)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=11, frameon=True)

    # 🌟 修复关键点 2：显式微调四周留白比例（特别是 top 增加到 0.85），完美解决标题被截断问题
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.10, right=0.90)
    plt.savefig(OUTPUT_FOLDER / "1_核心性能雷达对比图.png", dpi=300)
    plt.close()
    print(f"✅ 雷达对比图已保存至：{OUTPUT_FOLDER / '1_核心性能雷达对比图.png'}\n")


# -------------------------- 5. 全局热力图模块（保持不变） --------------------------
def plot_scaled_heatmap():
    print("🌡️ 开始生成 16模型 × 31原始指标 的全局综合热力图...")
    norm_matrix = np.zeros_like(median_matrix)
    for j in range(median_matrix.shape[1]):
        col = median_matrix[:, j]
        c_min, c_max = np.min(col), np.max(col)
        if c_max - c_min > 0:
            if "IFA" in REAL_METRICS[j] or REAL_METRICS[j] == "pf":
                norm_matrix[:, j] = (c_max - col) / (c_max - c_min)
            else:
                norm_matrix[:, j] = (col - c_min) / (c_max - c_min)
        else:
            norm_matrix[:, j] = 1.0

    fig, ax = plt.subplots(figsize=(24, 11))
    im = ax.imshow(norm_matrix, cmap="RdYlBu_r", aspect="auto")

    ax.set_xticks(range(len(REAL_METRICS)))
    ax.set_yticks(range(len(all_methods)))
    ax.set_xticklabels(REAL_METRICS, rotation=45, ha="right", rotation_mode="anchor", fontsize=10, fontweight="bold")
    ax.set_yticklabels(all_methods, fontsize=11, fontweight="bold")

    for i in range(len(all_methods)):
        for j in range(len(REAL_METRICS)):
            val = median_matrix[i, j]
            text_str = f"{val:.1f}" if val > 1.5 else f"{val:.3f}"
            color_sel = "white" if norm_matrix[i, j] > 0.82 or norm_matrix[i, j] < 0.18 else "black"
            ax.text(j, i, text_str, ha="center", va="center", fontsize=8.5, color=color_sel)

    group_sizes = [len([m for m in all_methods if method_to_group[m] == g]) for g in METHOD_CLASSIFICATION]
    sep_positions = [sum(group_sizes[:i]) - 0.5 for i in range(1, len(group_sizes)) if group_sizes[i - 1] > 0]
    for pos in sep_positions:
        ax.axvline(x=pos, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

    plt.title("Overall 31-Metric Performance Heatmap (Original Metric Names Matrix)", fontsize=14, fontweight="bold",
              pad=25)

    plt.subplots_adjust(bottom=0.22, top=0.90, left=0.10, right=0.95)
    plt.savefig(OUTPUT_FOLDER / "2_全局综合性能热力图.png", dpi=300)
    plt.close()
    print(f"✅ 全局综合热力图已成功保存至：{OUTPUT_FOLDER / '2_全局综合性能热力图.png'}\n")


if __name__ == "__main__":
    plot_single_grouped_bar()
    plot_radar_chart()
    plot_scaled_heatmap()
    print("🚀 所有修改全部完美生效！雷达图标题已被推至安全显示区域。")