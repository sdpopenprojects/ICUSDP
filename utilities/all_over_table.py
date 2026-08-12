import os
import glob
import pandas as pd
import numpy as np

# ==================== 路径与配置区域 ====================
INPUT_DIR = r"E:\ICUSDP-main\ICUSDP-main\new2\result"

# 💡【修改 1】：更新输出文件夹名称，区分无 MUSDP_1 版本
OUTPUT_DIR = r"E:\ICUSDP-main\ICUSDP-main\new2\all_over"

# 💡【修改 2】：目标模型列表（已移除 MUSDP_1，保留 KMedoids 与 MUSDP，共 17 个模型）
MODEL_COLUMNS = [
    'ICUSDP', 'ONE', 'CLA', 'CLAMI', 'MUSDP', 'MD', 'MU', 'SC',
    'TCL', 'TCLP', 'KMedoids', 'DT', 'GBM', 'linearSVM', 'LR', 'RF', 'XGBoost'
]

# MUSDP 专属的 13 个指标名称映射
MUSDP_13_METRICS = [
    'AUC', 'g_mean', 'precision', 'recall', 'pf',
    'F1', 'MCC', 'Popt', 'cErecall', 'cEprecision',
    'cEfmeasure', 'cPMI', 'cIFA'
]

# 固定 28 个 JIRA 项目的标准输出顺序
PROJECT_ORDER = [
    'activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
    'derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1',
    'groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2',
    'hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2',
    'hive-0.10.0', 'hive-0.12.0', 'hive-0.9.0',
    'jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1',
    'lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1',
    'wicket-1.3.0-beta2', 'wicket-1.3.0-incubating-beta-1', 'wicket-1.5.3'
]


# =======================================================

def standardize_project_name(proj):
    """
    【项目名称标准化】
    1. 剥离项目名称自带的 '.csv' 尾缀
    2. 统一转小写，移除横杠、下划线和点号差异
    """
    if pd.isna(proj):
        return ""
    s = str(proj).lower().strip()
    if s.endswith(".csv"):
        s = s[:-4]
    return s.replace("-", "").replace("_", "").replace(".", "").strip()


def extract_model_name(file_name):
    """
    文件名精准切片提取，锁定模型列名
    """
    name = file_name.lower().strip()

    # 1. 剪掉尾部后缀
    if name.endswith(".csv"):
        name = name[:-4]
    if name.endswith("_results"):
        name = name[:-8]

    # 2. 剪掉头部前缀
    if name.startswith("result_intc_"):
        name = name[12:]
    elif name.startswith("result_"):
        name = name[7:]
    elif name.startswith("all_result_"):
        name = name[11:]

    raw_extracted = name.strip()

    # 💡【修改 3】：清除 MUSDP_1 映射，保留 KMedoids 和 MUSDP 提取逻辑
    if raw_extracted == "linearsvm": return "linearSVM"
    if raw_extracted in ["kmedoids", "kmedoid"]: return "KMedoids"
    if raw_extracted in ["musdp_2", "musdp2", "musdp"]: return "MUSDP"

    # 3. 严格等值比对
    for col in MODEL_COLUMNS:
        if raw_extracted == col.lower():
            return col

    return file_name


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: {OUTPUT_DIR}")

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print(f"❌ 错误: 在路径 {INPUT_DIR} 中未找到任何 CSV 文件！")
        return

    # 嵌套字典: { 指标名: { 标准化项目名: { 模型名: 数值 } } }
    all_data = {}

    print("================== 1. 开始解析原始 CSV 文件 ==================")
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        model_name = extract_model_name(file_name)

        # 不在目标模型列表里的文件直接跳过
        if model_name not in MODEL_COLUMNS:
            continue

        try:
            # 针对 MUSDP 的特殊 13 列做定制化读取，常规方法走常规分支
            if 'MUSDP' in model_name:
                test_df = pd.read_csv(file_path, nrows=2)
                if test_df.shape[1] == 13:
                    df = pd.read_csv(file_path, header=None)
                    df.columns = MUSDP_13_METRICS
                    # 如果只有 28 行，且第一列不是项目名，则自动对齐 28 个项目名
                    if len(df) == 28 and df.index.dtype != 'object':
                        df.index = PROJECT_ORDER
                else:
                    df = pd.read_csv(file_path, index_col=0)
            else:
                df = pd.read_csv(file_path, index_col=0)

            df.index = df.index.astype(str).str.strip()
            df.columns = df.columns.astype(str).str.strip()

            for metric in df.columns:
                if 'time' in metric.lower():
                    continue

                if metric not in all_data:
                    all_data[metric] = {}

                # 遍历原始 CSV 中的所有项目行
                for raw_project in df.index:
                    std_proj = standardize_project_name(raw_project)

                    if std_proj not in all_data[metric]:
                        all_data[metric][std_proj] = {}

                    val = df.loc[raw_project, metric]

                    # 💡【修改 4】：针对 MUSDP 的 IFA 相关指标，进行全量减 1 处理
                    if model_name == 'MUSDP' and 'IFA' in metric:
                        val = val - 1

                    all_data[metric][std_proj][model_name] = val

            print(f"匹配成功 -> 标准列: [{model_name}] <- 源文件: {file_name}")
        except Exception as e:
            print(f"❌ 读取文件失败 {file_name}: {e}")

    if not all_data:
        print("❌ 未提取到任何有效的性能指标数据，程序终止。")
        return

    print("\n================== 2. 开始分别生成独立的指标 Excel 表格 ==================")

    file_count = 0
    for metric_name in sorted(all_data.keys()):
        metric_dict = all_data[metric_name]
        rows = []

        # 按照标准物理项目顺序构建输出行
        for proj in PROJECT_ORDER:
            row_dict = {'Project': proj}

            std_proj_key = standardize_project_name(proj)
            proj_scores = metric_dict.get(std_proj_key, {})

            for model in MODEL_COLUMNS:
                # 若某个模型没有当前指标，自动填为 np.nan
                row_dict[model] = proj_scores.get(model, np.nan)
            rows.append(row_dict)

        df_metric_table = pd.DataFrame(rows)

        # 安全清洗指标文件名
        safe_metric_name = "".join(
            [c for c in metric_name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).strip()
        metric_file_path = os.path.join(OUTPUT_DIR, f"{safe_metric_name}.xlsx")

        df_metric_table.to_excel(metric_file_path, index=False)
        file_count += 1

        # 显示包含模型的提示信息
        musdp_exist = df_metric_table['MUSDP'].notna().any()
        kmedoids_exist = df_metric_table['KMedoids'].notna().any()

        info_str = f"MUSDP:{'√' if musdp_exist else '×'} | KMedoids:{'√' if kmedoids_exist else '×'}"

        print(f"表格生成成功 -> [{file_count}]: {safe_metric_name}.xlsx ({info_str})")

    print("\n==============================================================")
    print(f"🎉 处理完成！数据表已更新（包含 17 个模型：含 KMedoids 与 MUSDP）。")
    print(f"📂 所有指标 Excel 已输出至: {OUTPUT_DIR}")
    print("==============================================================")


if __name__ == "__main__":
    main()