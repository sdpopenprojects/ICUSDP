import os
import pandas as pd

# =====================================================
# 路径配置
# =====================================================
input_dir = r"E:\ICUSDP\INTC\ICUSDP\W_C_test\visual\visual_data"
base_output = r"E:\ICUSDP\INTC\ICUSDP\W_C_test\visual\SKESD"

output_unsup = os.path.join(base_output, "dataall_unsupervised1")
os.makedirs(output_unsup, exist_ok=True)

# =====================================================
# 标准 32 个指标定义（适用于 ICUSDP 及大部分对比方法）
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
# MUSDP 专属列名映射定义（共 13 列）
# 提示：请根据你的 MUSDP 实际 CSV 列顺序修改下方列表中的指标名字
# =====================================================
musdp_metrics = [
    'AUC', 'g_mean', 'precision', 'recall', 'pf',
    'F1', 'MCC', 'Popt', 'cErecall', 'cEprecision',
    'cEfmeasure', 'cPMI', 'cIFA'
]

# =====================================================
# 无监督方法列表（用 MUSDP 替换了 KMedoids）
# =====================================================
unsupervised_methods = [
    'ICUSDP',
    'MUSDP',
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
# 读取所有无监督方法的数据
# =====================================================
all_method_data = {}

for method in unsupervised_methods:
    file_path = os.path.join(input_dir, f"all_result_{method}.csv")

    if not os.path.exists(file_path):
        print(f"Warning: 跳过不存在的文件 {file_path}")
        continue

    print("Reading:", file_path)
    df = pd.read_csv(file_path, header=None)

    # 针对 MUSDP 进行单独的列名映射
    if method == 'MUSDP':
        if df.shape[1] != len(musdp_metrics):
            raise ValueError(f"MUSDP 的列数为 {df.shape[1]}，与预设的 {len(musdp_metrics)} 不符！")
        df.columns = musdp_metrics
    else:
        if df.shape[1] != len(metrics):
            raise ValueError(f"{method} 的列数为 {df.shape[1]}，应为 {len(metrics)}。")
        df.columns = metrics

    all_method_data[method] = df

print("\n所有无监督方法数据读取完成。\n")

# =====================================================
# 生成无监督整合结果数据
# =====================================================
for metric in metrics:

    # 动态筛选包含当前指标的方法（若 MUSDP 无此指标则自动跳过）
    current_methods = [
        m for m in unsupervised_methods
        if m in all_method_data and metric in all_method_data[m].columns
    ]

    df_unsup = pd.DataFrame()
    for method in current_methods:
        df_unsup[method] = all_method_data[method][metric]

    # 保存文件
    df_unsup.to_csv(
        os.path.join(output_unsup, f"all_results_{metric}.csv"),
        index=False,
        header=False
    )

    print(f"{metric:<18}  √  (包含 {len(current_methods)} 个方法)")

print("\n========================================")
print("无监督数据整合生成完成！")
print("========================================")
print("输出目录:", output_unsup)