import os
import pandas as pd
import numpy as np

# =====================================================
# 输入路径
# =====================================================

input_dir = r"E:\ICUSDP-main\ICUSDP-main\new2\allresult"

# =====================================================
# 输出路径
# =====================================================

base_output = r"E:\ICUSDP-main\ICUSDP-main\new2"

output_all = os.path.join(base_output, "dataall")
output_unsup = os.path.join(base_output, "dataall_unsupervised")
output_sup = os.path.join(base_output, "dataall_supervised")

os.makedirs(output_all, exist_ok=True)
os.makedirs(output_unsup, exist_ok=True)
os.makedirs(output_sup, exist_ok=True)

# =====================================================
# 全部方法（加入 MUSDP）
# =====================================================

methods = [
    'ICUSDP',
    'KMedoids',
    'MUSDP',  # 💡 新增 MUSDP
    'MU',
    'MD',
    'SC',
    'TCL',
    'TCLP',
    'CLA',
    'CLAMI',
    'ONE',
    'DT',
    'RF',
    'GBM',
    'XGBoost',
    'LR',
    'linearSVM'
]

# =====================================================
# 无监督方法（含 ICUSDP 与 MUSDP）
# =====================================================

unsupervised_methods = [
    'ICUSDP',
    'KMedoids',
    'MUSDP',  # 💡 新增 MUSDP
    'MU',
    'MD',
    'SC',
    'TCL',
    'TCLP',
    'CLA',
    'CLAMI',
    'ONE'
]

# =====================================================
# 监督方法（含 ICUSDP）
# =====================================================

supervised_methods = [
    'ICUSDP',
    'DT',
    'RF',
    'GBM',
    'XGBoost',
    'LR',
    'linearSVM'
]

# =====================================================
# 标准 32 个指标列表
# =====================================================

metrics = [
    'precision', 'recall', 'pf', 'F1', 'AUC',
    'g_measure', 'g_mean', 'bal', 'MCC', 'accuracy',
    'Popt',
    'cErecall', 'cEprecision', 'cEfmeasure',
    'cMCC', 'cPMI', 'cIFA', 'cPCI',
    'c_ROI_PII', 'c_ROI_PCI', 'ceIFA',
    'mRecall', 'mPrecision', 'mfmeasure',
    'mMCC', 'mPMI', 'mIFA', 'mPCI',
    'm_ROI_PII', 'm_ROI_PCI', 'meIFA',
    'time'
]

# 💡【修改点 1】：定义 MUSDP 专属的 13 个指标顺序列表
musdp_metrics = [
    'AUC', 'g_mean', 'precision', 'recall', 'pf',
    'F1', 'MCC', 'Popt', 'cErecall', 'cEprecision',
    'cEfmeasure', 'cPMI', 'cIFA'
]

# =====================================================
# 读取所有方法
# =====================================================

all_method_data = {}

for method in methods:

    file_path = os.path.join(
        input_dir,
        f"all_result_{method}.csv"
    )

    print("Reading:", file_path)

    df_raw = pd.read_csv(file_path, header=None)

    # 💡【修改点 2】：单独处理 MUSDP（13列指标与 IFA-1 逻辑）
    if method == 'MUSDP':
        if df_raw.shape[1] != len(musdp_metrics):
            raise ValueError(
                f"MUSDP 的列数为 {df_raw.shape[1]}，应为 {len(musdp_metrics)}。"
            )

        # 绑定 MUSDP 的 13 个列名
        df_raw.columns = musdp_metrics

        # 💡【修改点 3】：把 MUSDP 的 IFA (对应 cIFA) 列的值减去 1
        if 'cIFA' in df_raw.columns:
            df_raw['cIFA'] = df_raw['cIFA'] - 1

        # 构建符合 32 个指标的全量 DataFrame（缺失指标填 NaN）
        df = pd.DataFrame(index=df_raw.index, columns=metrics)
        for col in musdp_metrics:
            df[col] = df_raw[col]

    else:
        # 其他标准 32 列方法正常校验与映射
        if df_raw.shape[1] != len(metrics):
            raise ValueError(
                f"{method} 的列数为 {df_raw.shape[1]}，应为 {len(metrics)}。"
            )
        df = df_raw.copy()
        df.columns = metrics

    all_method_data[method] = df

print("\n所有方法读取与预处理完成。\n")

# =====================================================
# 生成数据
# =====================================================

for metric in metrics:

    # -------------------------------
    # 全部方法
    # -------------------------------

    df_all = pd.DataFrame()

    for method in methods:
        df_all[method] = all_method_data[method][metric]

    df_all.to_csv(
        os.path.join(output_all, f"all_results_{metric}.csv"),
        index=False,
        header=False
    )

    # -------------------------------
    # 无监督
    # -------------------------------

    df_unsup = pd.DataFrame()

    for method in unsupervised_methods:
        df_unsup[method] = all_method_data[method][metric]

    df_unsup.to_csv(
        os.path.join(output_unsup, f"all_results_{metric}.csv"),
        index=False,
        header=False
    )

    # -------------------------------
    # 监督
    # -------------------------------

    df_sup = pd.DataFrame()

    for method in supervised_methods:
        df_sup[method] = all_method_data[method][metric]

    df_sup.to_csv(
        os.path.join(output_sup, f"all_results_{metric}.csv"),
        index=False,
        header=False
    )

    print(f"{metric:<18}  √")

print("\n========================================")
print("全部数据生成完成！")
print("========================================")
print(output_all)
print(output_unsup)
print(output_sup)