import csv
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from algorithms.InterpreableClustering_V1 import InterpretableClustering
from algorithms.labelingCluster import labelCluster
from utilities import performanceMeasure, rankMeasurev2
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X, LOC, n_class, v_lambda, max_iters, save_path, project_name, model_name, randseed, cluster_method='sc'):
    print(f"{project_name}: -> {model_name} ({cluster_method}) Round {randseed + 1} Start!")


    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X, randseed)
    test_LOC = LOC[test_idx]


    feature_names = X[0].columns.values
    test_X = preprocessing.scale(test_data[0])
    n_feas = test_X.shape[1]

    start = time.perf_counter()


    model = InterpretableClustering(
        n_clusters=n_class,
        hidden_dims=[n_feas * 2, n_feas],
        latent_dim=n_feas,
        lambda_ce=v_lambda,
        cluster_type=cluster_method,
        random_state=randseed
    )


    clus_label = model.fit_predict(test_X, max_iters=max_iters)
    t = time.perf_counter() - start


    report = model.get_interpretability_report(feature_names=feature_names)
    # fres_dir = create_dir(os.path.join(save_path, model_name, "reports"))
    # save_results_pickle(os.path.join(fres_dir, project_name), report)



    predict_y = labelCluster(test_X, clus_label)


    precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC, accuracy = performanceMeasure.get_measure(test_label, predict_y)


    (Popt, cErecall, cEprecision, cEfmeasure, cMCC, cPMI, cIFA, cPCI, c_ROI_PII, c_ROI_PCI, ceIFA,
    mRecall, mPrecision, mfmeasure, mMCC, mPMI, mIFA, mPCI, m_ROI_PII, m_ROI_PCI, meIFA) = rankMeasurev2.rank_measure(predict_y, test_LOC, test_label)


    measure = [

        precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC, accuracy,


        Popt, cErecall, cEprecision, cEfmeasure, cMCC, cPMI, cIFA, cPCI, c_ROI_PII, c_ROI_PCI, ceIFA,


        mRecall, mPrecision, mfmeasure, mMCC, mPMI, mIFA, mPCI, m_ROI_PII, m_ROI_PCI, meIFA,


        t
    ]

    res_path = create_dir(os.path.join(save_path, model_name + '_results'))
    save_results(os.path.join(res_path, project_name), measure)

    return report


if __name__ == '__main__':
    warnings.filterwarnings('ignore')


    # methods_to_run = ['sc', 'gmm', 'agglomerative', 'kmedoids']
    methods_to_run = ['kmeans']
    Reps = 100
    n_class = 2
    v_lambda = 0.1
    max_iters = 10
    data_dir = '../data/'

    for cluster_method in methods_to_run:
        current_save_path = f'../new/noFS'
        model_name = f'INTC_{cluster_method.upper()}'

        project_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])

        for file_name in project_files:
            project_name_base = file_name[:-4]
            data = pd.read_csv(os.path.join(data_dir, file_name))


            if 'CountLineCode' in data.columns:
                LOC = data['CountLineCode']
            else:
                LOC = data.iloc[:, 0]


            X_features = data.iloc[:, :-1]
            y = data.iloc[:, -1].copy()
            y[y > 1] = 1


            X_data = [X_features, y]

            print(f"\n>>> runing: {cluster_method} | project: {project_name_base} | : {X_features.shape[1]}")


            project_reports = []

            for loop in range(Reps):

                rep = run_(X_data, LOC, n_class, v_lambda, max_iters,
                           current_save_path, project_name_base, model_name,
                           loop, cluster_method=cluster_method)
                project_reports.append(rep)


            fres_dir = create_dir(os.path.join(current_save_path, model_name, "reports"))
            save_results_pickle(os.path.join(fres_dir, project_name_base), project_reports)

    print("\n run over！")