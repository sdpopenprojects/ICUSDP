import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.AutoSpearman import AutoSpearman


current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


def run_md(X_data, LOC, save_path, project_name, model_name, randseed):


    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_data, randseed)


    if isinstance(LOC, pd.Series):
        test_LOC = LOC.iloc[test_idx].values
    else:
        test_LOC = LOC[test_idx]


    test_X = preprocessing.scale(test_data[0])

    start = time.perf_counter()


    score = test_LOC.copy()

    t = time.perf_counter() - start


    predict_y = np.where(score >= np.median(score), 1, 0)

    y_true_int = np.array(test_label).astype(int)
    y_pred_int = np.array(predict_y).astype(int)


    m1 = performanceMeasure.get_measure(y_true_int, y_pred_int)


    res_rank = rankMeasurev2.rank_measure(score, test_LOC, test_label)


    m2 = res_rank[:11]
    m3 = res_rank[11:]


    measure = list(m1) + list(m2) + list(m3) + [t]


    res_path = create_dir(os.path.join(save_path, model_name + '_results'))
    save_results(os.path.join(res_path, project_name), measure)


if __name__ == '__main__':

    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore')


    Reps = 100
    data_dir = '../data/'

    current_save_path = f'../new/result_USDP/MD/'
    model_name = 'INTC_MD'

    project_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for file_name in project_files:
        project_name_base = file_name[:-4]
        data = pd.read_csv(os.path.join(data_dir, file_name))


        if 'CountLineCode' in data.columns:
            LOC = data['CountLineCode']
        elif 'loc' in data.columns:
            LOC = data['loc']
        else:
            LOC = data.iloc[:, 0]


        X_features = data.iloc[:, :-1]
        y = data.iloc[:, -1].copy()
        y[y > 1] = 1


        print(f">>> runing {project_name_base}  (AutoSpearman)...")
        X_selected = AutoSpearman(X_features)
        print(f"   : {X_features.shape[1]} -> {X_selected.shape[1]}")


        features_dir = create_dir(os.path.join(current_save_path, 'features'))
        fs_file = os.path.join(features_dir, f"{project_name_base}_features.csv")
        with open(fs_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Selected_Features'])
            for feat in X_selected.columns.tolist():
                writer.writerow([feat])

        X_data = [X_selected, y]

        print(f"\n>>> runing MD  | : {project_name_base} | : {X_selected.shape[1]}")


        for loop in range(Reps):
            run_md(X_data, LOC, current_save_path, project_name_base, model_name, loop)
            if (loop + 1) % 10 == 0:
                print(f"  {project_name_base}: Round {loop + 1} completed.")

    print("\n run over！")