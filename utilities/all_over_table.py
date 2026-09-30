import os
import glob
import pandas as pd
import numpy as np


INPUT_DIR = r"E:\ICUSDP\INTC\ICUSDP-main\new\FS\dataprocess2\result"


OUTPUT_DIR = r"E:\ICUSDP\INTC\ICUSDP-main\new\FS\dataprocess2\all_over"


MODEL_COLUMNS = [
    'ICUSDP', 'ONE', 'CLA', 'CLAMI', 'MD', 'MU', 'SC',
    'TCL', 'TCLP', 'KMedoids', 'DT', 'GBM', 'linearSVM', 'LR', 'RF', 'XGBoost'
]


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

    if pd.isna(proj):
        return ""
    s = str(proj).lower().strip()
    if s.endswith(".csv"):
        s = s[:-4]
    return s.replace("-", "").replace("_", "").replace(".", "").strip()


def extract_model_name(file_name):

    name = file_name.lower().strip()


    if name.endswith(".csv"):
        name = name[:-4]
    if name.endswith("_results"):
        name = name[:-8]


    if name.startswith("result_intc_"):
        name = name[12:]
    elif name.startswith("result_"):
        name = name[7:]
    elif name.startswith("all_result_"):
        name = name[11:]

    raw_extracted = name.strip()


    if raw_extracted == "linearsvm": return "linearSVM"
    if raw_extracted in ["kmedoids", "kmedoid"]: return "KMedoids"


    for col in MODEL_COLUMNS:
        if raw_extracted == col.lower():
            return col

    return file_name


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f": {OUTPUT_DIR}")

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print(f" {INPUT_DIR} no CSV ！")
        return


    all_data = {}


    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        model_name = extract_model_name(file_name)


        if model_name not in MODEL_COLUMNS:
            continue

        try:

            df = pd.read_csv(file_path, index_col=0)

            df.index = df.index.astype(str).str.strip()
            df.columns = df.columns.astype(str).str.strip()

            for metric in df.columns:
                if 'time' in metric.lower():
                    continue

                if metric not in all_data:
                    all_data[metric] = {}


                for raw_project in df.index:
                    std_proj = standardize_project_name(raw_project)

                    if std_proj not in all_data[metric]:
                        all_data[metric][std_proj] = {}

                    val = df.loc[raw_project, metric]

                    all_data[metric][std_proj][model_name] = val

            print(f": [{model_name}] <- : {file_name}")
        except Exception as e:
            print(f"fail {file_name}: {e}")

    if not all_data:
        print("❌ no")
        return



    file_count = 0
    for metric_name in sorted(all_data.keys()):
        metric_dict = all_data[metric_name]
        rows = []


        for proj in PROJECT_ORDER:
            row_dict = {'Project': proj}

            std_proj_key = standardize_project_name(proj)
            proj_scores = metric_dict.get(std_proj_key, {})

            for model in MODEL_COLUMNS:

                row_dict[model] = proj_scores.get(model, np.nan)
            rows.append(row_dict)

        df_metric_table = pd.DataFrame(rows)


        safe_metric_name = "".join(
            [c for c in metric_name if c.isalpha() or c.isdigit() or c in (' ', '_', '-')]).strip()
        metric_file_path = os.path.join(OUTPUT_DIR, f"{safe_metric_name}.xlsx")

        df_metric_table.to_excel(metric_file_path, index=False)
        file_count += 1


        kmedoids_exist = df_metric_table['KMedoids'].notna().any()
        info_str = f"KMedoids:{'√' if kmedoids_exist else '×'}"

        print(f"-> [{file_count}]: {safe_metric_name}.xlsx ({info_str})")




if __name__ == "__main__":
    main()