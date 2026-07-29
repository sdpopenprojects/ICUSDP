import os
import pandas as pd

# =====================================================
# 输入路径（16个方法结果）
# =====================================================

input_dir = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\visual\visual_data"

# =====================================================
# 输出路径
# =====================================================

base_output = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\visual\SKE"

output_all = os.path.join(base_output, "dataall")
output_unsup = os.path.join(base_output, "dataall_unsupervised")
output_sup = os.path.join(base_output, "dataall_supervised")

os.makedirs(output_all, exist_ok=True)
os.makedirs(output_unsup, exist_ok=True)
os.makedirs(output_sup, exist_ok=True)

# =====================================================
# 全部方法
# =====================================================

methods = [
    'ICUSDP',
    'KMedoids',
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
# 无监督方法（含 ICUSDP）
# =====================================================

unsupervised_methods = [
    'ICUSDP',
    'KMedoids',
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
# 指标
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

    df = pd.read_csv(
        file_path,
        header=None
    )

    if df.shape[1] != len(metrics):
        raise ValueError(
            f"{method} 的列数为 {df.shape[1]}，应为 {len(metrics)}。"
        )

    df.columns = metrics

    all_method_data[method] = df

print("\n所有方法读取完成。\n")

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