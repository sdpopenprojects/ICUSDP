import csv
import multiprocessing
import os
import sys
import time
import warnings

# sys.path.append(os.path.dirname(sys.path[0]))
import pandas as pd
import optuna

from functools import partial
from sklearn import preprocessing

from algorithms.InterpretableClustering import InterpretableClustering
from algorithms.labelingCluster import labelCluster, labelCluster_v2
from utilities import performanceMeasure, rankMeasure
from utilities.AutoSpearman import AutoSpearman
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X, LOC, n_class, v_lambda, max_iters, save_path, project_name, model_name, randseed):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')
    feature_names = X[0].columns.values
    n_feas = len(feature_names)

    # 数据划分
    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X, randseed)
    # LOC = np.array(test_data[0].iloc[:, 27]) # CountLineCode
    # LOC = test_data[0]['CountLineCode']
    test_LOC = LOC[test_idx]

    # test_data = [AutoSpearman(x) for x in test_data]
    # test_data = [preprocessing.scale(x) for x in test_data]
    test_data = preprocessing.scale(test_data[0])

    # running time
    start = time.perf_counter()

    # hidden_dims = [130, 65]
    # latent_dim = 65

    # 初始化INTC模型
    model = InterpretableClustering(n_clusters=n_class, hidden_dims=[n_feas * 2, n_feas], latent_dim=n_feas, clf=model_name,
                                    lambda_ce=v_lambda, random_state=randseed)
    clus_label = model.fit_predict(test_data, max_iters=max_iters)

    end = time.perf_counter()
    t = end - start

    # Interpretability analysis 生成解释性报告
    report = model.get_interpretability_report(feature_names=feature_names)
    fres = create_dir(save_path + 'INTC_' + model_name)
    save_results_pickle(fres + project_name, report)

    # 两种标注方法
    # labeling clustering
    predict_y1 = labelCluster(test_data, clus_label)
    predict_y2 = labelCluster_v2(clus_label)

    # # calculate non-effort-aware classification measure
    precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC = performanceMeasure.get_measure(test_label, predict_y1)
    # # calculate cost-effectiveness measures
    Popt, Erecall, Eprecision, Efmeasure, PMI, IFA = rankMeasure.rank_measure(predict_y1, test_LOC, test_label)

    measure = [precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC, Popt, Erecall, Eprecision, Efmeasure, PMI,
               IFA, t]
    fres = create_dir(save_path + 'INTC_' + model_name + '_label1')
    save_results(fres + project_name, measure)

    # # calculate non-effort-aware classification measure
    precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC = performanceMeasure.get_measure(test_label, predict_y2)
    # # calculate cost-effectiveness measures
    Popt, Erecall, Eprecision, Efmeasure, PMI, IFA = rankMeasure.rank_measure(predict_y2, test_LOC, test_label)

    measure2 = [precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC, Popt, Erecall, Eprecision, Efmeasure, PMI,
                IFA, t]
    fres = create_dir(save_path + 'INTC_'+ model_name + '_label2')
    save_results(fres + project_name, measure2)


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    save_path = '../result_20251016/INTC-classifiers_norandom/'
    # model_name = 'INTC_iter30'
    model_names = ['RF', 'GBM', 'XGBoost', 'DT'] # , 'LR', 'linearSVM'
    Reps = 100
    n_class = 2
    v_lambda = 0.1
    max_iters = 10

    project_names = sorted(os.listdir('../data/'))
    path = os.path.abspath('../data/')
    pro_num = len(project_names)
    # max_cpu = multiprocessing.cpu_count()

    for model_name in model_names:
        for i in range(pro_num):  # pro_num
            # for i in range(pro_num-1, 10, -1):
            project_name = project_names[i]
            file = os.path.join(path, project_name)
            data = pd.read_csv(file)
            project_name = project_name[:-4]
            # feature_names = data.columns.values[:-1]

            # JIRA
            LOC = data['CountLineCode']

            # Promise
            # LOC = data['loc']
            # data = data.iloc[:, :-1]

            # ck
            # LOC = data['loc']

            # construct data
            X = [data.iloc[:, 0:65]]
            y = data.iloc[:, -1]
            y[y > 1] = 1

            # feature selection -> correlation analysis and redundancy analysis
            # X = [AutoSpearman(x) for x in X]
            # feas = []
            # for x in X:
            #     feas.extend(x.columns.values)
            #
            # fres = create_dir(save_path)
            # with open(fres + 'features_mv.csv', 'a', newline='') as f:
            #     # f.write(np.array2string(feas))
            #     # f.write('\n')
            #     writer = csv.writer(f)
            #     writer.writerow(feas)

            X.append(y)

            for loop in range(Reps):
                run_(X, LOC, n_class, v_lambda, max_iters, save_path, project_name, model_name, loop)

            # if i in [6, 7, 8]: # for camel project with larger instances
            #     max_cpu = 16
            # else:
            #     max_cpu = multiprocessing.cpu_count()

            # with multiprocessing.Pool(max_cpu) as p:
            #     p.map(partial(run_, X, LOC, n_class, v_lambda, save_path, project_name, model_name), range(Reps))
            #     p.close()
            #     p.join()

    print('done!')
