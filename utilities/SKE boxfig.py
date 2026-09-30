import glob
import os
import warnings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings('ignore', category=UserWarning)


METRIC_NAME_MAP = {

    'precision': 'Precision',
    'recall': 'Recall',
    'pf': 'Probability of False Alarm (PF)',
    'F1': 'F1-Measure',
    'AUC': 'AUC',
    'g_measure': 'G-Measure',
    'g_mean': 'G-Mean',
    'bal': 'Balance(Bal)',
    'MCC': 'MCC',
    'accuracy': 'Accuracy',

    'Popt': '$P_{opt}$',

    'cErecall': 'Recall@20%LOC',
    'cEprecision': 'Precision@20%LOC',
    'cEfmeasure': 'F-measure@20%LOC',
    'cMCC': 'MCC@20%LOC',
    'cPMI': 'PMI@20%LOC',
    'cIFA': 'IFA',
    'cPCI': 'PCI@20%LOC',
    'c_ROI_PII': 'ROI@20%LOC',
    'c_ROI_PCI': 'ROI_PCI@20%LOC',
    'ceIFA': 'eIFA',

    'mRecall': 'Recall@20%Modules',
    'mPrecision': 'Precision@20%Modules',
    'mfmeasure': 'F-measure@20%Modules',
    'mMCC': 'MCC@20%Modules',
    'mPMI': 'PMI@20%Modules',
    'mIFA': 'IFA@20%Modules',
    'mPCI': 'PCI@20%Modules',
    'm_ROI_PII': 'ROI_PII@20%Modules',
    'm_ROI_PCI': 'ROI@20%Modules',
    'meIFA': 'eIFA',

    'time': 'Execution Time (s)',
}

INPUT_DIR = r'E:\ICUSDP\INTC\ICUSDP-main\new\FS\dataprocess2\all_over'
SK_SUPERVISED_DIR = r'E:\ICUSDP\INTC\ICUSDP-main\new\FS\dataprocess2\SDP'
SK_UNSUPERVISED_DIR = r'E:\ICUSDP\INTC\ICUSDP-main\new\FS\dataprocess2\USDP'
OUTPUT_DIR = r'E:\ICUSDP\INTC\ICUSDP-main\new\FS\dataprocess2\SDP VS USDP'
MAIN_MODEL = 'ICUSDP'


def get_sk_groups(metric_name, is_supervised=True):
    base_dir = SK_SUPERVISED_DIR if is_supervised else SK_UNSUPERVISED_DIR
    sk_file = os.path.join(base_dir, f'NPSKESD_group_{metric_name}_CLF.csv')

    if not os.path.exists(sk_file):
        print(
            f'⚠️ : {sk_file}'
        )
        return None

    try:
        sk_df = pd.read_csv(sk_file, header=None)
        if len(sk_df) < 2:
            print(f'⚠️: {sk_file}')
            return None

        model_row = sk_df.iloc[-2].dropna().values
        group_row = sk_df.iloc[-1].dropna().values

        group_map = {}
        for model_name, group_num in zip(model_row, group_row):
            clean_model = str(model_name).strip().replace('"', '').replace("'", '')
            group_map[clean_model] = int(float(group_num))

        return group_map
    except Exception as e:
        print(f'❌ error {metric_name}: {e}')
        return None


def plot_single_category_sk(
        plot_df, metric_name, target_cols, category_label, is_supervised, save_dir
):

    TITLE_FONT_SIZE = 18
    LABEL_FONT_SIZE = 16
    YTICKS_FONT_SIZE = 14
    XTICKS_FONT_SIZE = 16
    SK_NUM_FONT_SIZE = 14


    sub_cols_raw = [
        col for col in (['ICUSDP'] + target_cols) if col in plot_df.columns
    ]
    sub_cols_raw = list(dict.fromkeys(sub_cols_raw))

    if len(sub_cols_raw) <= 1:
        return

    if 'linearSVM' in plot_df.columns:
        plot_df = plot_df.rename(columns={'linearSVM': 'SVM'})
    sub_cols_raw = ['SVM' if x == 'linearSVM' else x for x in sub_cols_raw]

    sk_group_map = get_sk_groups(metric_name, is_supervised=is_supervised)

    if sk_group_map and 'linearSVM' in sk_group_map:
        sk_group_map['SVM'] = sk_group_map['linearSVM']

    model_ranks = {}
    model_sort_weights = {}

    for model in sub_cols_raw:
        if sk_group_map and model in sk_group_map:
            group_num = sk_group_map[model]
            model_ranks[model] = group_num
            is_main = 0 if model == 'ICUSDP' else 1
            model_sort_weights[model] = (-group_num, is_main)
        else:
            model_ranks[model] = 99
            is_main = 0 if model == 'ICUSDP' else 1
            model_sort_weights[model] = (-99, is_main)

    sub_cols = sorted(sub_cols_raw, key=lambda x: model_sort_weights[x])

    if 'linearSVM' in plot_df.columns:
        cleaned_plot_df = plot_df.rename(columns={'linearSVM': 'SVM'})
    else:
        cleaned_plot_df = plot_df.copy()

    sub_df = cleaned_plot_df[sub_cols]

    sk_colors = {
        1: '#E64B35',
        2: '#4DBBD5',
        3: '#00A087',
        4: '#F39B7F',
        5: '#8491B4',
        6: '#91D1C2',
        7: '#DC0000',
        8: '#7E6148',
        9: '#B09C85',
    }

    valid_ranks = [r for r in model_ranks.values() if r != 99]
    max_rank = max(valid_ranks) if valid_ranks else 1

    palette_colors = []
    for model in sub_cols:
        rank_no = model_ranks[model]
        if isinstance(rank_no, int) and rank_no != 99:
            display_no = max_rank - rank_no + 1
            palette_colors.append(sk_colors.get(display_no, '#CCCCCC'))
        else:
            palette_colors.append('#CCCCCC')


    fig, ax = plt.subplots(figsize=(12, 7.5))


    sns.boxplot(
        data=sub_df,
        palette=palette_colors,
        linewidth=1.2,
        width=0.5,
        ax=ax,
        flierprops=dict(
            marker='o',
            markersize=4.5,
            markerfacecolor='none',
            markeredgecolor='#444444',
            alpha=0.7,
        ),
    )


    ymax = sub_df.max().max()
    ymin = sub_df.min().min()
    y_range = ymax - ymin if (ymax - ymin) != 0 else 1
    offset = y_range * 0.02


    for i, model in enumerate(sub_cols):
        rank_no = model_ranks[model]
        if rank_no != 99:
            display_no = max_rank - rank_no + 1
            label_text = f'#{display_no}'
            color = 'red' if display_no == 1 else 'black'
        else:
            label_text = '?'
            color = 'black'

        ax.text(
            i,
            ymax + offset,
            label_text,
            ha='center',
            va='bottom',
            fontsize=SK_NUM_FONT_SIZE,
            fontweight='bold',
            color=color,
        )


    ax.set_ylim(ymin - offset, ymax + y_range * 0.15)


    formal_metric_name = METRIC_NAME_MAP.get(metric_name, metric_name)
    custom_title = f'Comparison of {formal_metric_name} in {category_label} Methods'
    plt.title(custom_title, fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=18)


    plt.xticks(rotation=45, ha='right', fontsize=XTICKS_FONT_SIZE)
    plt.yticks(fontsize=YTICKS_FONT_SIZE)
    plt.ylabel(formal_metric_name, fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)


    plt.subplots_adjust(top=0.88, bottom=0.25, left=0.12, right=0.95)


    img_path_png = os.path.join(
        save_dir, f'{metric_name}_{category_label}_SK_boxplot.png'
    )
    plt.savefig(img_path_png, dpi=300)

    img_path_pdf = os.path.join(
        save_dir, f'{metric_name}_{category_label}_SK_boxplot.pdf'
    )
    plt.savefig(img_path_pdf, format='pdf', bbox_inches='tight')

    plt.close()


def plot_boxplot_split(df, metric_name, save_dir):
    plot_df = df.copy()
    if 'Project' in plot_df.columns:
        plot_df = plot_df.drop(columns=['Project'])

    supervised_models = ['DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM']


    unsupervised_models = [
        'KMedoids',
        'MU',
        'MD',
        'SC',
        'TCL',
        'TCLP',
        'CLA',
        'CLAMI',
        'ONE',
    ]

    plot_single_category_sk(
        plot_df, metric_name, supervised_models, 'Supervised', True, save_dir
    )
    plot_single_category_sk(
        plot_df, metric_name, unsupervised_models, 'Unsupervised', False, save_dir
    )


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f' {OUTPUT_DIR}')

    excel_files = glob.glob(os.path.join(INPUT_DIR, '*.xlsx'))
    if not excel_files:
        print(f'❌  {INPUT_DIR} ')
        return

    total_files = len(excel_files)


    for idx, file_path in enumerate(excel_files, 1):
        file_name = os.path.basename(file_path)
        metric_name = os.path.splitext(file_name)[0]

        try:
            df = pd.read_excel(file_path)
            plot_boxplot_split(df, metric_name, OUTPUT_DIR)
            print(
                f'[{idx}/{total_files}] '
                f' PDF）：[{metric_name}]'
            )
        except Exception as e:
            print(f'❌  {file_name}: {e}')


if __name__ == '__main__':
    main()