import csv
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from algorithms.InterpreableClustering_V1_replace1 import InterpretableClustering
from algorithms.labelingCluster import labelCluster
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap
from utilities.AutoSpearman import AutoSpearman


def run_(X_data, LOC, n_class, v_lambda, max_iters, save_path, project_name, model_name, randseed, cluster_method='sc',
         learner_type='VAE'):
    print(f"{project_name}: -> {model_name} ({cluster_method} + {learner_type}) Round {randseed + 1} Start!")


    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X_data, randseed)
    test_LOC = LOC.iloc[test_idx].values


    feature_names = X_data[0].columns.values
    test_X = preprocessing.scale(test_data[0])
    n_feas = test_X.shape[1]

    start = time.perf_counter()


    model = InterpretableClustering(
        n_clusters=n_class,
        # hidden_dims=[n_feas * 2, n_feas],
        # latent_dim=n_feas,
        hidden_dims=[128, 64],
        latent_dim=32,
        lambda_ce=v_lambda,
        cluster_type=cluster_method,
        feature_learner=learner_type,
        random_state=randseed
    )


    clus_label = model.fit_predict(test_X, max_iters=max_iters)
    t = time.perf_counter() - start


    report = model.get_interpretability_report(feature_names=feature_names)
    fres_dir = create_dir(os.path.join(save_path, model_name, "reports"))
    save_results_pickle(os.path.join(fres_dir, project_name), report)


    predict_y = labelCluster(test_X, clus_label)


    m1 = performanceMeasure.get_measure(test_label, predict_y)


    res_rank = rankMeasurev2.rank_measure(predict_y, test_LOC, test_label)


    m2 = res_rank[:11]
    m3 = res_rank[11:]


    measure = list(m1) + list(m2) + list(m3) + [t]


    res_path = create_dir(os.path.join(save_path, model_name + '_results'))
    save_results(os.path.join(res_path, project_name), measure)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')


    methods_to_run = ['kmeans']
    learners_to_run = ['AE', 'DAE']

    Reps = 100
    n_class = 2
    v_lambda = 0.1
    max_iters = 10
    data_dir = '../data/'

    current_save_path = '../result_Dissicussion/replace1/'


    fs_save_path = os.path.join(current_save_path, 'feas')
    if not os.path.exists(fs_save_path):
        os.makedirs(fs_save_path)

    for learner in learners_to_run:
        for cluster_method in methods_to_run:

            model_name = f'INTC_{cluster_method.upper()}_{learner}'

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
                y[y > 1] = 1  # 转换为二分类标签


                print(f">>>  {project_name_base}  (AutoSpearman)...")
                X_features = AutoSpearman(X_features)


                selected_features = X_features.columns.tolist()
                fs_file = os.path.join(fs_save_path, f"{project_name_base}_features.csv")
                with open(fs_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Selected_Features'])
                    for feat in selected_features:
                        writer.writerow([feat])

                X_data = [X_features, y]

                print(f"\n>>> featuer learning={learner} + cluster={cluster_method} | project: {project_name_base} | : {X_features.shape[1]}")

                for loop in range(Reps):
                    run_(X_data, LOC, n_class, v_lambda, max_iters,
                         current_save_path, project_name_base, model_name,
                         loop, cluster_method=cluster_method, learner_type=learner)

    print("\n[SUCCESS] VAE/AE/DAE run over！")