import csv
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing


current_file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_file_path))
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from algorithms.InterpreableClustering_V1_CAE import InterpretableClustering
from algorithms.labelingCluster import labelCluster
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.AutoSpearman import AutoSpearman



def run_unsupervised_cae_iteration(X_package, LOC, n_class, max_iters, save_path, project_name, model_name, randseed,
                                   cluster_method):

    print(f"{project_name}: -> {model_name} ({cluster_method}) Round {randseed + 1} Start!")


    train_res, train_label, test_res, test_label, train_idx, test_idx = outofsample_bootstrap(X_package, randseed)


    test_X_df = test_res[0] if isinstance(test_res, list) else test_res

    if isinstance(LOC, pd.Series):
        t_loc = LOC.iloc[test_idx].values
    else:
        t_loc = LOC[test_idx]

    feature_names = X_package[0].columns.values


    test_X_scaled = preprocessing.scale(test_X_df.values)
    n_feas = test_X_scaled.shape[1]

    start_time = time.perf_counter()

    try:
        model = InterpretableClustering(
            n_clusters=n_class,
            hidden_dims=[128, 64],
            latent_dim=32,
            clf='DT',
            epochs=200,
            lr=1e-3,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lambda_recon=0.7,
            random_state=randseed
        )


        test_predict_labels = model.fit_predict(test_X_scaled, max_iters=max_iters, cluster_method=cluster_method)


        y_test_predict = np.array(test_predict_labels).flatten()
        labeled_test_cluster = labelCluster(test_X_scaled, y_test_predict)

        exec_time = time.perf_counter() - start_time


        y_true = np.array(test_label).astype(int)
        y_pred = np.array(labeled_test_cluster).astype(int)


        m1 = performanceMeasure.get_measure(y_true, y_pred)


        res_rank = rankMeasurev2.rank_measure(y_pred.astype(float), t_loc, test_label)
        m2 = res_rank[:11]
        m3 = res_rank[11:]


        full_measures = list(m1) + list(m2) + list(m3) + [exec_time]


        res_dir = create_dir(os.path.join(save_path, f"{model_name}_results"))
        save_results(os.path.join(res_dir, project_name), full_measures)


        try:
            report = model.get_report(feature_names)
            report_dir = create_dir(os.path.join(save_path, model_name, "reports"))
            save_results_pickle(os.path.join(report_dir, project_name), report)
        except Exception as report_e:
            print(f"  Warning saving report for {project_name}: {report_e}")

    except Exception as e:
        print(f"  !!! Iteration {randseed} Error in {model_name}: {e}")
        import traceback
        traceback.print_exc()



if __name__ == '__main__':
    warnings.filterwarnings('ignore')


    methods_to_run = ['kmeans']
    Reps = 100
    n_class = 2
    max_iters = 10

    data_dir = '../data/'
    save_path_root = f'../disscussion/replace1/CAE/clustering'


    fs_save_path = os.path.join(save_path_root, 'E:../disscussion/replace1/CAE/feas')
    if not os.path.exists(fs_save_path):
        os.makedirs(fs_save_path)

    project_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

    for cluster_method in methods_to_run:
        model_name = f'INTC_{cluster_method.upper()}'
        print(f"\n{'#' * 70}\n: {model_name}\n{'#' * 70}")

        for file_name in project_files:
            p_name = file_name[:-4]
            raw_df = pd.read_csv(os.path.join(data_dir, file_name))


            if 'CountLineCode' in raw_df.columns:
                LOC_series = raw_df['CountLineCode']
            elif 'loc' in raw_df.columns:
                LOC_series = raw_df['loc']
            else:
                LOC_series = raw_df.iloc[:, 0]


            X_raw = raw_df.iloc[:, :-1]
            y_series = raw_df.iloc[:, -1].copy()
            y_series[y_series > 1] = 1


            print(f">>> {p_name}  (AutoSpearman)...")
            X_raw = AutoSpearman(X_raw)


            selected_features = X_raw.columns.tolist()
            fs_file = os.path.join(fs_save_path, f"{p_name}_features.csv")
            with open(fs_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Selected_Features'])
                for feat in selected_features:
                    writer.writerow([feat])

            X_package = [X_raw, y_series]

            print(f"\n>>>  {p_name} | : {X_raw.shape[1]} |  {Reps} ...")
            for randseed in range(Reps):
                run_unsupervised_cae_iteration(
                    X_package=X_package,
                    LOC=LOC_series,
                    n_class=n_class,
                    max_iters=max_iters,
                    save_path=save_path_root,
                    project_name=p_name,
                    model_name=model_name,
                    randseed=randseed,
                    cluster_method=cluster_method
                )

    print("\n save", save_path_root)