# 基准对比实验，用于将传统的变分自编码器（VAE）+逻辑回归（LR） 方法作为基准
import os
import time
import warnings

import numpy as np
import pandas as pd
import optuna
import torch

from sklearn import preprocessing

from algorithms.AutoEncoder import VariationalAutoencoder, LR
from algorithms.Classifiers2 import OptimizingCLF
from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results, save_results_pickle
from utilities.bootstrapCV import outofsample_bootstrap

# VAE+LR两阶段方法:先VAE，后LR
def callVAE(train_data, train_label, test_data, args):
    input_dim = args['input_dim']
    hidden_dims = args['hidden_dims']
    latent_dim = args['latent_dim']
    # hidden_dim = args['hidden_dims'][0]
    # latent_dim = args['hidden_dims'][1]

    vaemodel = VariationalAutoencoder(input_dim, hidden_dims, latent_dim).to(args['device'])
    vaemodel.train_vae(train_data)
    vae_train_features, _ = vaemodel.extract_embedded_features(train_data)

    # vaemodel.train_vae(test_data)
    vae_test_features, _ = vaemodel.extract_embedded_features(test_data)

    # vae_train_features = preprocessing.scale(vae_train_features)
    # vae_test_features = preprocessing.scale(vae_test_features)

    lr_model = LR(input_dim=vae_train_features.shape[1], num_classes=2).to(args['device'])
    lr_model.fit_model(vae_train_features, train_label)
    y_pred = lr_model.predict_model(vae_test_features)

    return y_pred, lr_model


def run_(X, LOC, save_path, project_name, model_name, randseed, args):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')
    feature_names = X[0].columns.values
    n_feas = len(feature_names)

    train_data, train_label, test_data, test_label, train_idx, test_idx = outofsample_bootstrap(X, randseed)

    # LOC = np.array(test_data[0].iloc[:, 27]) # CountLineCode
    # LOC = test_data[0]['CountLineCode']
    test_LOC = LOC[test_idx]
    test_data = preprocessing.scale(test_data[0])

    train_data = preprocessing.scale(train_data[0])

    # running time
    start = time.perf_counter()

    # train_data = np.c_[train_data[0], train_label]
    # train_data = np.unique(train_data, axis=0)  # delete repetitive rows

    # train_data, train_label, val_data, val_label, _, _ = outofsample_bootstrap(pd.DataFrame(train_data), randseed)
    # train_data = preprocessing.scale(train_data)
    # val_data = preprocessing.scale(val_data)

    # model = OptimizingCLF(train_data, train_label, val_data, val_label, classifier=model_name)
    # clf = model.getOptCLF()
    #
    # clf.fit(train_data, train_label)
    # predict_y = clf.predict(test_data)

    predict_y, model = callVAE(train_data, train_label, test_data, args=args)

    end = time.perf_counter()
    t = end - start

    # Interpretability analysis
    report = model.get_interpretability_report(feature_names=feature_names)
    fres = create_dir(save_path + model_name)
    save_results_pickle(fres + project_name, report)

    # # calculate non-effort-aware classification measure
    precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC = performanceMeasure.get_measure(test_label, predict_y)
    # # calculate cost-effectiveness measures
    Popt, Erecall, Eprecision, Efmeasure, PMI, IFA = rankMeasure.rank_measure(predict_y, test_LOC, test_label)

    measure = [precision, recall, pf, F1, AUC, g_measure, g_mean, bal, MCC, Popt, Erecall, Eprecision, Efmeasure, PMI,
               IFA, t]
    fres = create_dir(save_path + model_name)
    save_results(fres + project_name, measure)


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings('ignore')

    save_path = '../result_20251016/VAE/'

    model_names = ['VAE_v1']
    Reps = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("We're using =>", device)

    args = {'epochs': 200, 'batch_size': 512, 'input_dim': 65, 'output_dim': 2, 'device': device,
            'hidden_dims': [130, 65], 'latent_dim': 65, 'n_layers': 3, 'lr': 1e-3, 'dropout': 0.1, 'weight_decay': 1e-4}

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
                run_(X, LOC, save_path, project_name, model_name, loop, args)

            # if i in [6, 7, 8]: # for camel project with larger instances
            #     max_cpu = 16
            # else:
            #     max_cpu = multiprocessing.cpu_count()

            # with multiprocessing.Pool(max_cpu) as p:
            #     p.map(partial(run_, X, LOC, n_class, v_lambda, save_path, project_name, model_name), range(Reps))
            #     p.close()
            #     p.join()

    print('done!')
