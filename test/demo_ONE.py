import os
import csv
import pandas as pd
import numpy as np
from utilities import performanceMeasure, rankMeasurev2
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.File import create_dir, save_results
from utilities.AutoSpearman import AutoSpearman


from algorithms.algorithm_ONE import ONE


def run_ONE_on_JIRA(data_path, save_path, reps=100):

    res_root = create_dir(os.path.join(save_path, 'ONE_Results'))


    fs_save_path = create_dir(os.path.join(save_path, 'features'))

    project_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])

    for file_name in project_files:
        project_name = file_name[:-4]


        csv_path = os.path.join(data_path, file_name)
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            separator = ';' if ';' in first_line else ','

        data = pd.read_csv(csv_path, sep=separator)
        data.columns = data.columns.str.strip()


        bug_col_exact = None
        for col in data.columns:
            if col.lower() in ['label', 'bugs']:
                bug_col_exact = col
                break

        if bug_col_exact is None:
            raise KeyError(f" in{file_name} no 'label'or'bugs' 。")

        y_raw = data[bug_col_exact].copy()
        y_numeric = pd.to_numeric(y_raw, errors='coerce').fillna(0)
        y = np.where(y_numeric >= 1, 1, 0)


        bug_columns = [col for col in data.columns if 'bug' in col.lower() or 'label' in col.lower()]
        exclude_cols = bug_columns + ['classname']
        X_raw = data.drop(columns=[col for col in exclude_cols if col in data.columns])
        X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(float)


        print(f">>>  {project_name} (AutoSpearman)...")
        X_selected = AutoSpearman(X_raw)
        print(f"   : {X_raw.shape[1]} -> {X_selected.shape[1]}")


        fs_file = os.path.join(fs_save_path, f"{project_name}_features.csv")
        with open(fs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Selected_Features'])
            for feat in X_selected.columns.tolist():
                writer.writerow([feat])

        X_with_y = X_selected.copy()
        X_with_y['target_label'] = y


        if 'CountLineCode' in data.columns:
            LOC = data['CountLineCode']
        elif 'CountLine' in data.columns:
            LOC = data['CountLine']
        elif 'loc' in data.columns:
            LOC = data['loc']
        elif 'linesAddedUntil' in data.columns:
            LOC = data['linesAddedUntil']
        else:
            LOC = X_raw.iloc[:, 0]
            LOC = data.iloc[:, 0]

        print(f">>> Project {project_name}: Processing with Feature Selection, using {X_selected.shape[1]} features...")

        for r in range(reps):

            _, _, _, test_label, _, test_idx = outofsample_bootstrap(X_with_y, r)

            t_label = test_label.values if hasattr(test_label, 'values') else test_label
            t_loc = LOC.iloc[test_idx].values


            final_data = ONE(t_loc, t_label)


            final_data['score'] = final_data['predict_label']


            final_data = final_data.sort_values(by=['original_idx'], ascending=[True])

            one_scores = final_data['score'].values
            one_predict_y = final_data['predict_label'].values

            y_true_int = np.array(t_label).astype(int)
            y_pred_int = np.array(one_predict_y).astype(int)


            m1 = performanceMeasure.get_measure(y_true_int, y_pred_int)

            res_rank = rankMeasurev2.rank_measure(one_scores, t_loc, t_label)

            m2 = res_rank[:11]
            m3 = res_rank[11:]


            measure = list(m1) + list(m2) + list(m3) + [0.0]


            save_results(os.path.join(res_root, project_name), measure)

        print(f"Project {project_name}: All {reps} rounds completed.")


if __name__ == '__main__':
    data_input_path = '../data/'
    result_output_path = '../new/result_USDP/ONE/'

    run_ONE_on_JIRA(data_input_path, result_output_path, reps=100)