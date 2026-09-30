import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing


current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from algorithms.TCL import TCL
from algorithms.TCLP import TCLP
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.AutoSpearman import AutoSpearman



def run_tclp_iteration(X_package, LOC, save_path, project_name, model_name, randseed):

    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)

    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]


    test_X_raw = test_data[0]
    test_X_scaled = preprocessing.scale(test_X_raw)


    test_X_df = pd.DataFrame(test_X_scaled, columns=test_X_raw.columns)

    start_time = time.perf_counter()

    try:

        clus_label, metric_threshold = TCL(test_X_df)


        predict_y = TCLP(test_X_df, clus_label, metric_threshold)


        pred_scores = predict_y.astype(float)

        exec_time = time.perf_counter() - start_time


        y_true = np.array(test_label).astype(int)
        y_pred = np.array(predict_y).astype(int)


        m1 = performanceMeasure.get_measure(y_true, y_pred)


        res_rank = rankMeasurev2.rank_measure(pred_scores, t_loc, test_label)
        m2 = res_rank[:11]
        m3 = res_rank[11:]


        full_measures = list(m1) + list(m2) + list(m3) + [exec_time]


        res_dir = create_dir(os.path.join(save_path, model_name + '_results'))
        save_results(os.path.join(res_dir, project_name), full_measures)

    except Exception as e:
        print(f"  Iteration {randseed} Error: {e}")



if __name__ == '__main__':
    warnings.filterwarnings('ignore')


    Reps = 100
    data_dir = '../data/'
    save_path_root = '../new/result_USDP/TCLP/'
    model_tag = 'INTC_TCLP'

    project_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for file in project_list:
        p_name = file[:-4]
        raw_df = pd.read_csv(os.path.join(data_dir, file))


        loc_candidates = ['CountLineCode', 'loc', 'LOC']
        loc_col = next((c for c in loc_candidates if c in raw_df.columns), raw_df.columns[0])
        LOC_series = raw_df[loc_col]

        label_col = raw_df.columns[-1]
        y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)
        X_raw = raw_df.iloc[:, :-1]


        print(f">>>  {p_name}  (AutoSpearman)...")
        X_selected = AutoSpearman(X_raw)
        print(f"   : {X_raw.shape[1]} -> {X_selected.shape[1]}")


        features_dir = create_dir(os.path.join(save_path_root, 'features'))
        fs_file = os.path.join(features_dir, f"{p_name}_features.csv")
        with open(fs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Selected_Features'])
            for feat in X_selected.columns.tolist():
                writer.writerow([feat])

        X_package = [X_selected, y_series]

        print(f"\n>>>  {model_tag} | : {p_name} | : {X_selected.shape[1]}")


        for r in range(Reps):
            run_tclp_iteration(X_package, LOC_series, save_path_root, p_name, model_tag, r)
            if (r + 1) % 10 == 0:
                print(f"  Status: {r + 1}/{Reps} rounds.")

    print("\n TCLP run over！")