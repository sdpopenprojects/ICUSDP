import os
import pandas as pd
import numpy as np
from pathlib import Path

# -------------------------- 1. 路径精确锁定 --------------------------
# 严格按照指示锁定输入与输出路径
INPUT_FOLDER = Path(r"F:\ICUSDP\INTC\ICUSDP\W_C_test\visual\visual_data")
OUTPUT_FOLDER = Path(r"F:\ICUSDP\INTC\ICUSDP\W_C_test\table")
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

# 16 个模型 2800 行全量原始大表的文件名映射
FILE_MAPPING = {
    "ONE": "all_result_ONE.csv",
    "DT": "all_result_DT.csv",
    "GBM": "all_result_GBM.csv",
    "linearSVM": "all_result_linearSVM.csv",
    "LR": "all_result_LR.csv",
    "RF": "all_result_RF.csv",
    "XGBoost": "all_result_XGBoost.csv",
    "CLA": "all_result_CLA.csv",
    "CLAMI": "all_result_CLAMI.csv",
    "KMedoids": "all_result_KMedoids.csv",
    "MD": "all_result_MD.csv",
    "MU": "all_result_MU.csv",
    "SC": "all_result_SC.csv",
    "TCL": "all_result_TCL.csv",
    "TCLP": "all_result_TCLP.csv",
    "ICUSDP": "all_result_ICUSDP.csv"
}

# 严格对应 2800 行大表中的 31 个全指标顺序
REAL_METRICS = [
    "precision", "recall", "pf", "F1", "AUC",
    "g_measure", "g_mean", "bal", "MCC", "accuracy",
    "Popt", "cErecall", "cEprecision", "cEfmeasure", "cMCC",
    "cPMI", "cIFA", "cPCI", "c_ROI_PII", "c_ROI_PCI",
    "ceIFA", "mRecall", "mPrecision", "mfmeasure", "mMCC",
    "mPMI", "mIFA", "mPCI", "m_ROI_PII", "m_ROI_PCI",
    "meIFA"
]

# 28 个标准的项目名称顺序（严格进行顺序切片对齐）
PROJECT_NAMES = [
    "activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.8.0",
    "camel-1.4.0", "camel-1.6.0", "derby-10.2.1.6", "derby-10.3.1.4", "derby-10.5.1.1",
    "geronimo-1.1", "hbase-0.94.0", "hbase-0.95.0", "hive-0.9.0", "hive-0.10.0",
    "hive-0.12.0", "jruby-1.1", "jruby-1.5.0", "jruby-1.7.0", "lucene-2.0",
    "lucene-2.2", "lucene-2.4", "mahout-0.5", "mahout-0.6", "mahout-0.7",
    "poi-3.0", "wicket-1.3.0", "wicket-1.5.3"
]

SUPERVISED_MODELS = ["ONE", "DT", "GBM", "linearSVM", "LR", "RF", "XGBoost"]
UNSUPERVISED_MODELS = ["CLA", "CLAMI", "KMedoids", "MD", "MU", "SC", "TCL", "TCLP"]
OUR_MODEL = "ICUSDP"
RUNS_PER_PROJECT = 100  # 每个项目独立运行的轮数


# -------------------------- 2. 核心大表切片与双轨计算 --------------------------
def generate_true_statistical_tables():
    # 存储结构：{ metric_name: { model_name: { 'mean': [...], 'median': [...] } } }
    processed_data = {m: {} for m in REAL_METRICS}

    print(f"📖 正在从输入路径读取并切片 2800 行原始大表...")
    for model_name, filename in FILE_MAPPING.items():
        file_path = INPUT_FOLDER / filename
        if not file_path.exists():
            print(f"⚠️ 提示: 未在输入路径找到大表 {filename}，已跳过该模型。")
            continue

        # 原始 2800 行数据无表头，切分前 31 列指标
        df = pd.read_csv(file_path, header=None).iloc[:, :len(REAL_METRICS)]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 校验行数安全性
        expected_rows = len(PROJECT_NAMES) * RUNS_PER_PROJECT
        if df.shape[0] < expected_rows:
            print(f"❌ 错误: {filename} 的行数 ({df.shape[0]}) 不足 {expected_rows} 行，无法安全重组项目数据！")
            continue

        for m_name in REAL_METRICS:
            processed_data[m_name][model_name] = {'mean': [], 'median': []}

        # 每 100 行还原一个项目在 100 轮中的真实 Mean & Median
        for p_idx, p_name in enumerate(PROJECT_NAMES):
            start_row = p_idx * RUNS_PER_PROJECT
            end_row = start_row + RUNS_PER_PROJECT
            project_chunk = df.iloc[start_row:end_row].values  # 捕获 (100, 31) 矩阵

            p_means = np.mean(project_chunk, axis=0)
            p_medians = np.median(project_chunk, axis=0)

            for m_idx, m_name in enumerate(REAL_METRICS):
                processed_data[m_name][model_name]['mean'].append(p_means[m_idx])
                processed_data[m_name][model_name]['median'].append(p_medians[m_idx])

    print(f"📊 正在组装对比表并导出至：{OUTPUT_FOLDER} ...")
    # -------------------------- 3. 双轨分离组装输出 --------------------------
    for m_name in REAL_METRICS:
        if OUR_MODEL not in processed_data[m_name]:
            continue

        for stat_type in ['mean', 'median']:
            suffix = "平均数" if stat_type == 'mean' else "中位数"

            # 3.1 监督阵营对比表（监督 vs ICUSDP）
            sup_list = [m for m in SUPERVISED_MODELS if m in processed_data[m_name]] + [OUR_MODEL]
            if len(sup_list) > 1:
                sup_df = pd.DataFrame({m: processed_data[m_name][m][stat_type] for m in sup_list}, index=PROJECT_NAMES)
                sup_df.index.name = "Project"
                # 计算 28 个项目聚合后的宏观 Mean / Median
                sup_df.loc["Mean"] = sup_df.mean()
                sup_df.loc["Median"] = sup_df.median()
                # 针对软件工程学术规范，IFA 类保留1位小数，其余保留3位
                sup_df = sup_df.round(1) if "IFA" in m_name else sup_df.round(3)
                sup_df.to_csv(OUTPUT_FOLDER / f"Table_{m_name}_vs_Supervised_{suffix}.csv", encoding="utf-8-sig")

            # 3.2 无监督阵营对比表（无监督 vs ICUSDP）
            unsup_list = [m for m in UNSUPERVISED_MODELS if m in processed_data[m_name]] + [OUR_MODEL]
            if len(unsup_list) > 1:
                unsup_df = pd.DataFrame({m: processed_data[m_name][m][stat_type] for m in unsup_list},
                                        index=PROJECT_NAMES)
                unsup_df.index.name = "Project"
                unsup_df.loc["Mean"] = unsup_df.mean()
                unsup_df.loc["Median"] = unsup_df.median()

                unsup_df = unsup_df.round(1) if "IFA" in m_name else unsup_df.round(3)
                unsup_df.to_csv(OUTPUT_FOLDER / f"Table_{m_name}_vs_Unsupervised_{suffix}.csv", encoding="utf-8-sig")

    print(f"🎉 完美的学术三线表基础数据已全部产出！请前往 {OUTPUT_FOLDER} 查看。")


if __name__ == "__main__":
    generate_true_statistical_tables()