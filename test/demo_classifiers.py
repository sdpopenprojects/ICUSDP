import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import optuna
from sklearn import preprocessing


current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from algorithms.Classifiers2 import OptimizingCLF
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap



def run_supervised_iteration(X_package, LOC, save_path, project_name, model_name, randseed):

    train_res, train_label, test_res, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)


    train_X_df = train_res[0] if isinstance(train_res, list) else train_res
    test_X_df = test_res[0] if isinstance(test_res, list) else test_res

    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]

    feature_names = X_package[0].columns.values


    train_X_scaled = preprocessing.scale(train_X_df.values)
    test_X_scaled = preprocessing.scale(test_X_df.values)

    start_time = time.perf_counter()

    try:

        sub_res, sub_label, val_res, val_label, _, _ = outofsample_bootstrap(
            [pd.DataFrame(train_X_scaled), train_label], randseed)


        actual_sub_train_x = sub_res[0].values if isinstance(sub_res, list) else sub_res.values
        actual_val_x = val_res[0].values if isinstance(val_res, list) else val_res.values


        opt_model = OptimizingCLF(actual_sub_train_x, sub_label, actual_val_x, val_label, classifier=model_name)
        clf = opt_model.getOptCLF()


        clf.fit(train_X_scaled, train_label)
        predict_y = clf.predict(test_X_scaled)


        if hasattr(clf, "predict_proba"):
            pred_scores = clf.predict_proba(test_X_scaled)[:, 1]
        else:
            pred_scores = predict_y.astype(float)

        exec_time = time.perf_counter() - start_time


        try:

            opt_model.clfmodel = clf


            report = opt_model.get_interpretability_report(feature_names=feature_names)


            fres_dir = create_dir(os.path.join(save_path, model_name, "reports"))
            save_results_pickle(os.path.join(fres_dir, project_name), report)

        except Exception as report_e:
            print(f"  Warning: Failed to generate interpretability report for {model_name}: {report_e}")

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
        print(f"  Iteration {randseed} Error in {model_name}: {e}")
        import traceback
        traceback.print_exc()



if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')


    Reps = 100
    model_names = ['DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM']
    # model_names = ['RF', 'GBM', 'XGBoost', 'LR', 'linearSVM']
    data_dir = r'../data/'
    save_path_root = r'..\result_SDP\supervised'

    project_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for model_tag in model_names:
        print(f"\n{'#' * 70}\n: {model_tag}\n{'#' * 70}")

        for file in project_list:
            p_name = file[:-4]
            raw_df = pd.read_csv(os.path.join(data_dir, file))


            loc_candidates = ['CountLineCode', 'loc', 'LOC']
            loc_col = next((c for c in loc_candidates if c in raw_df.columns), raw_df.columns[0])
            LOC_series = raw_df[loc_col]

            label_col = 'label' if 'label' in raw_df.columns else raw_df.columns[-1]
            y_series = raw_df[label_col].apply(lambda x: 1 if x > 0 else 0)


            X_raw = raw_df.drop(columns=[label_col]) if label_col in raw_df.columns else raw_df.iloc[:, :-1]


            X_package = [X_raw, y_series]


            print(f">>> run {model_tag} | project: {p_name} | : {X_raw.shape[1]}")
            for r in range(Reps):
                run_supervised_iteration(X_package, LOC_series, save_path_root, p_name, model_tag, r)
                if (r + 1) % 10 == 0:
                    print(f"  {p_name} ({model_tag}): {r + 1}/{Reps} ")

    print("\n run over！")