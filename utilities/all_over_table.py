import os
import glob
import pandas as pd
import numpy as np

# ==================== 路径与配置区域 ====================
INPUT_DIR = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\all_before"
OUTPUT_DIR = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\all_over"

# 严格对应你要求的列名和物理顺序（已去掉 VAE+SC, CAE+kmeans, CAE+SC，剩余 16 个模型）
MODEL_COLUMNS = [
    'ICUSDP', 'ONE', 'CLA', 'CLAMI', 'KMedoids', 'MD', 'MU', 'SC',
    'TCL', 'TCLP', 'DT', 'GBM', 'linearSVM', 'LR', 'RF', 'XGBoost'
]

# 固定 28 个 JIRA 项目的标准输出顺序（严格对应你模板表的字面样式）
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
    【核心修复函数】
    1. 强行剥离部分原始 CSV 中项目名称自带的 '.csv' 恶性尾缀！
    2. 统一转小写，移除横杠和下划线差异，实现多源数据的完美行对齐。
    """
    if pd.isna(proj):
        return ""
    s = str(proj).lower().strip()
    if s.endswith(".csv"):  # 专门切除像 activemq-5.0.0.csv 尾部的 .csv
        s = s[:-4]
    return s.replace("-", "").replace("_", "").replace(".", "").strip()


def extract_model_name(file_name):
    """
    文件名精准切片提取，确保模型列名大小写完美锁定
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

    raw_extracted = name.strip()

    # 特殊映射：只保留 linearSVM 的容错映射（删除了 VAE 和 CAE 的映射）
    if raw_extracted == "linearsvm": return "linearSVM"

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

        # 如果提取出的模型不在 16 个目标模型里，直接跳过不处理，防止脏数据混入
        if model_name not in MODEL_COLUMNS:
            continue

        try:
            df = pd.read_csv(file_path, index_col=0)
            df.index = df.index.str.strip()
            df.columns = df.columns.str.strip()

            for metric in df.columns:
                if 'time' in metric.lower():
                    continue

                if metric not in all_data:
                    all_data[metric] = {}

                # 遍历原始 CSV 中的所有项目行
                for raw_project in df.index:
                    # 将诸如 'activemq-5.0.0.csv' 转化为标准名进行数据聚合
                    std_proj = standardize_project_name(raw_project)

                    if std_proj not in all_data[metric]:
                        all_data[metric][std_proj] = {}

                    all_data[metric][std_proj][model_name] = df.loc[raw_project, metric]

            print(f"匹配成功 -> 标准列: [{model_name}] <- 源文件: {file_name}")
        except Exception as e:
            print(f"❌ 读取文件失败 {file_name}: {e}")

    if not all_data:
        print("❌ 未提取到任何有效的性能指标数据，程序终止。")
        return

    print("\n================== 2. 开始分别生成 34 个独立的指标 Excel 表格 ==================")

    file_count = 0
    for metric_name in sorted(all_data.keys()):
        metric_dict = all_data[metric_name]
        rows = []

        # 按照标准物理项目顺序构建输出行
        for proj in PROJECT_ORDER:
            row_dict = {'Project': proj}

            # 同样转换标准列表中的项目名，完美取出聚合的值
            std_proj_key = standardize_project_name(proj)
            proj_scores = metric_dict.get(std_proj_key, {})

            for model in MODEL_COLUMNS:
                row_dict[model] = proj_scores.get(model, np.nan)
            rows.append(row_dict)

        df_metric_table = pd.DataFrame(rows)

        # 安全清洗指标文件名
        safe_metric_name = "".join(
            [c for c in metric_name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).strip()
        metric_file_path = os.path.join(OUTPUT_DIR, f"{safe_metric_name}.xlsx")

        df_metric_table.to_excel(metric_file_path, index=False)
        file_count += 1
        print(f"表格生成成功 -> [{file_count}/34]: {safe_metric_name}.xlsx")

    print("\n==============================================================")
    print(f"🎉 优化完成！已剔除 VAE+SC、CAE+kmeans 和 CAE+SC。")
    print(f"📂 16 个核心模型的 34 个独立指标 Excel 已成功重新填满并输出！")
    print("==============================================================")


if __name__ == "__main__":
    main()