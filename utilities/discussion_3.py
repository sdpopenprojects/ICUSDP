import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_dir = r"E:\ICUSDP\INTC\ICUSDP-main\new\FS\disscussion\replace3"


project_list = [
      'activemq-5.0.0',
      'activemq-5.1.0',
      'activemq-5.2.0',
      'activemq-5.3.0',
      'activemq-5.8.0',
      'derby-10.2.1.6',
      'derby-10.3.1.4',
      'derby-10.5.1.1',
      'groovy-1_5_7',
      'groovy-1_6_beta1',
      'groovy-1_6_beta2',
      'hbase-0.94.0',
      'hbase-0.95.0',
      'hbase-0.95.2',
      'hive-0.10.0',
      'hive-0.12.0',
      'hive-0.9.0',
      'jruby-1.1',
      'jruby-1.4.0',
      'jruby-1.5.0',
      'jruby-1.7.0',
      'lucene-2.3.0',
      'lucene-2.9.0',
      'lucene-3.0.0',
      'lucene-3.1',
      'wicket-1.3.0-beta2',
      'wicket-1.3.0-beta1',
      'wicket-1.5.3',
]
reps = 100


metrics_config = {
    'MCC': 8,
    'F-measure@20%LOC': 13
}

project_median_records = []


search_pattern = os.path.join(data_dir, "all_result_*.csv")
file_paths = glob.glob(search_pattern)

if not file_paths:
    print(f"❌ Error: No files matching all_result_*.csv found in {data_dir}!")
else:
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            interpreter_name = file_name.split("_")[-2].upper()
        except Exception:
            interpreter_name = "Unknown"

        df_raw = pd.read_csv(file_path, header=None)

        for i, project_name in enumerate(project_list):
            start_idx = i * reps
            end_idx = start_idx + reps

            if start_idx < len(df_raw):
                df_chunk = df_raw.iloc[start_idx:end_idx]

                for metric_name, col_idx in metrics_config.items():
                    median_value = df_chunk.iloc[:, col_idx].median()

                    project_median_records.append({
                        'Project': project_name,
                        'Interpreter_Method': interpreter_name,
                        'Metric_Name': metric_name,
                        'Median_Score': median_value
                    })

    summary_df = pd.DataFrame(project_median_records)


    plt.style.use('seaborn-v0_8-whitegrid')


    fig, ax = plt.subplots(figsize=(9, 7.0))


    LABEL_FONT_SIZE = 22
    TICKS_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 15
    LEGEND_TITLE_SIZE = 16
    # =======================================================

    interpreter_order = ['DT', 'RF', 'GBM', 'XGBOOST']

    metric_order = ['MCC', 'F-measure@20%LOC']


    sns.boxplot(
        x='Metric_Name',
        y='Median_Score',
        hue='Interpreter_Method',
        order=metric_order,
        hue_order=interpreter_order,
        data=summary_df,
        palette='Set2',
        width=0.5,
        fliersize=3.5,
        linewidth=1.2,
        ax=ax
    )



    ax.set_xlabel('', labelpad=0)


    ax.set_ylabel('Values', fontsize=LABEL_FONT_SIZE, fontweight='bold', labelpad=10)


    ax.tick_params(axis='x', labelsize=TICKS_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICKS_FONT_SIZE)


    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    ax.set_ylim(bottom=0.0)


    legend_obj = ax.legend(
        title='Interpreter',
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        columnspacing=1.0,
        borderpad=0.3
    )


    for t in legend_obj.get_texts():
        if t.get_text() == 'XGBOOST':
            t.set_text('XGBoost')

    plt.tight_layout()


    save_jpg_path = os.path.join(data_dir, "generalization_classifier_overall_mcc_fmeasure.jpg")
    save_pdf_path = os.path.join(data_dir, "generalization_classifier_overall_mcc_fmeasure.pdf")


    try:
        plt.savefig(save_jpg_path, dpi=300, bbox_inches='tight')
        print(f": {save_jpg_path}")
    except PermissionError:
        print(f": {save_jpg_path}")


    try:
        plt.savefig(save_pdf_path, bbox_inches='tight')
        print(f": {save_pdf_path}")
    except PermissionError:
        alt_pdf_path = os.path.join(data_dir, "generalization_classifier_overall_mcc_fmeasure_new.pdf")
        plt.savefig(alt_pdf_path, bbox_inches='tight')
        print(f" {alt_pdf_path}")

    plt.close()

    print("\n" + "=" * 60)
    print("=" * 60)