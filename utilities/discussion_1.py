import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_dir = r"E:\ICUSDP\INTC\ICUSDP-main\new\disscussion\replace1"
# data_dir = r"F:\ICUSDP\INTC\ICUSDP\result_20260630_Discussion\replace3_1"


project_list = [
    'activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0',
    'camel-1.4.0', 'camel-1.6.0', 'derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1',
    'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3', 'poi-1.5', 'poi-2.0', 'poi-2.5', 'poi-3.0',
    'prop-1', 'prop-2', 'prop-3', 'prop-4', 'prop-5', 'tomcat', 'velocity-1.4', 'velocity-1.5',
    'velocity-1.6', 'wicket-1.3.0'
]
reps = 100


metrics_config = {
    'G-mean': 6,
    'MCC': 8,
    'F-measure@20%LOC': 13,
    'IFA': 16
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
            learner_name = file_name.split("_")[-2].upper()
        except Exception:
            learner_name = "Unknown"

        df_raw = pd.read_csv(file_path, header=None)

        for i, project_name in enumerate(project_list):
            start_idx = i * reps
            end_idx = start_idx + reps

            if start_idx < len(df_raw):
                df_chunk = df_raw.iloc[start_idx:end_idx]

                for metric_name, col_idx in metrics_config.items():
                    median_value = df_chunk.iloc[:, col_idx].median()

                    if metric_name == 'IFA':
                        median_value = np.log10(median_value + 1)

                    project_median_records.append({
                        'Project': project_name,
                        'Feature_Learner': learner_name,
                        'Metric_Name': metric_name,
                        'Median_Score': median_value
                    })

    summary_df = pd.DataFrame(project_median_records)


    plt.style.use('seaborn-v0_8-whitegrid')


    fig, ax = plt.subplots(figsize=(11.5, 7.0))


    LABEL_FONT_SIZE = 22
    TICKS_FONT_SIZE = 18
    LEGEND_FONT_SIZE = 15
    LEGEND_TITLE_SIZE = 16

    learner_order = ['VAE', 'AE', 'DAE', 'CAE']

    metric_order = ['G-mean', 'MCC', 'F-measure@20%LOC', 'IFA']


    sns.boxplot(
        x='Metric_Name',
        y='Median_Score',
        hue='Feature_Learner',
        order=metric_order,
        hue_order=learner_order,
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


    ax.legend(
        title='Feature Learner',
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        columnspacing=1.0,
        borderpad=0.3
    )

    plt.tight_layout()


    save_jpg_path = os.path.join(data_dir, "generalization_feature_learner_overall_vae_first.jpg")
    save_pdf_path = os.path.join(data_dir, "generalization_feature_learner_overall_vae_first.pdf")


    try:
        plt.savefig(save_jpg_path, dpi=300, bbox_inches='tight')
        print(f": {save_jpg_path}")
    except PermissionError:
        print(f": {save_jpg_path}")


    try:
        plt.savefig(save_pdf_path, bbox_inches='tight')
        print(f" : {save_pdf_path}")
    except PermissionError:
        alt_pdf_path = os.path.join(data_dir, "generalization_feature_learner_overall_gmean_new.pdf")
        plt.savefig(alt_pdf_path, bbox_inches='tight')
        print(f"re: {alt_pdf_path}")

    plt.close()

    print("\n" + "=" * 60)
    print("=" * 60)