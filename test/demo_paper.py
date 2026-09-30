import os
import sys
import pandas as pd
import numpy as np
import warnings


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

warnings.filterwarnings('ignore')

from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results


def run_effort_alignment_on_predictions(data_path, save_path, cutoff_pct=0.2):


    res_root = create_dir(os.path.join(save_path, f'Aligned_Results_Effort_{int(cutoff_pct * 100)}'))


    project_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])

    for file_name in project_files:
        project_name = file_name[:-4]
        file_full_path = os.path.join(data_path, file_name)

        print(f">>>> Processing Project: {project_name} with Effort Alignment...")


        df = pd.read_csv(file_full_path)


        df.columns = [c.strip() for c in df.columns]


        true_label_col = 'actualBugLabel' if 'actualBugLabel' in df.columns else 'label'
        loc_col = 'sloc' if 'sloc' in df.columns else 'loc'
        pred_value_col = 'predictedValue' if 'predictedValue' in df.columns else 'score'


        df['density'] = df[pred_value_col] / (df[loc_col] + 1e-8)


        df_sorted = df.sort_values(by=['density', loc_col], ascending=[False, True]).reset_index(drop=True)


        total_loc = df_sorted[loc_col].sum()
        df_sorted['cumsum_loc'] = df_sorted[loc_col].cumsum()


        df_sorted['aligned_predict_label'] = 0


        df_sorted.loc[df_sorted['cumsum_loc'] <= total_loc * cutoff_pct, 'aligned_predict_label'] = 1


        y_true = df_sorted[true_label_col].apply(lambda x: 1 if x > 0 else 0).values.astype(int)
        y_pred = df_sorted['aligned_predict_label'].values.astype(int)

        scores = df_sorted[pred_value_col].values.astype(float)
        locs = df_sorted[loc_col].values.astype(float)


        if len(np.unique(y_pred)) < 2:
            y_pred[0] = 1 - y_pred[0]


        try:
            m1 = performanceMeasure.get_measure(y_true, y_pred)
            res_rank = rankMeasure.rank_measure(scores, locs, y_true.astype(float))

            m2 = res_rank[:11]
            m3 = res_rank[11:]


            measure = list(m1) + list(m2) + list(m3) + [0.0]


            save_results(os.path.join(res_root, project_name), measure)
            print(f"✅ Project {project_name} processed successfully.")

        except Exception as e:
            print(f"❌ Project {project_name} failed to evaluate: {e}")
            continue

    print(f"\n✨ All intermediate prediction files have been aligned and evaluated!")


if __name__ == '__main__':

    prediction_data_dir = '..\ONE-result\cutoff0.2_exclude20\AEEEM'


    result_output_dir = '../result_ONE/Effort_Aligned_Table/'

    run_effort_alignment_on_predictions(prediction_data_dir, result_output_dir, cutoff_pct=0.2)