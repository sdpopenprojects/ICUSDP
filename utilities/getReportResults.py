import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # path = r"../result_20241008/plotfig/fs/"
    # path = r"E:/researchproject/unsupervised/ICSE2024/TR/round1/plot_results_fs/allResults_methods/"
    # path = r"F:/interpretability/code/INTC/result_20251016/clustering/INTC_K-means_label1/"
    path = r"F:/interpretability/code/INTC/result_20251016/clustering/INTC_K-means_label2/"

    files = [f for f in sorted(os.listdir(path + '/')) if f.endswith('.csv')]
    files_num = len(files)

    # methods = ['CLA', 'CLAMI', 'SC', 'ManualDown', 'ManualUp', 'LR', 'RF', 'DNN', 'MVKNN',
    #             'EASC', 'MUSDP_v1', 'MUSDP_v2']
    # file_idx = [0, 1, 11, 8, 9, 4, 10, 2, 7, 3, 5, 6]

    # methods = ['SC_C_v1', 'SC_P_v1', 'SC_O_v1', 'MUSDP_CP_v1', 'MUSDP_CO_v1', 'MUSDP_PO_v1', 'MUSDP_v1',
    #            'SC_C_v2', 'SC_P_v2', 'SC_O_v2', 'MUSDP_CP_v2', 'MUSDP_CO_v2', 'MUSDP_PO_v2', 'MUSDP_v2']
    # file_idx = [8, 12, 10, 2, 0, 4, 6, 9, 13, 11, 3, 1, 5, 7]

    # methods = ['LR', 'RF', 'DNN', 'EASC', 'MVLR', 'MVRF', 'MVDNN', 'MVEASC', 'MUSDP_v1', 'MUSDP_v2']
    # file_idx = [2, 9, 0, 1, 7, 8, 5, 6, 3, 4]

    # methods = ['LR', 'RF', 'DNN', 'MVKNN', 'EASC', 'LR*', 'RF*', 'DNN*', 'MVKNN*', 'EASC*', 'MUSDP_v1', 'MUSDP_v2']
    # file_idx = [5, 11, 1, 9, 3, 4, 10, 0, 8, 2, 6, 7]

    methods = ['CLA', 'CLAMI', 'SC', 'Kmedoids', 'ManualDown', 'ManualUp', 'TCL', 'TCLP', 'DNN', 'CNN', 'GRU', 'MVKNN',
                'EASC', 'MUSDP_v1', 'MUSDP_v2']
    file_idx = [1, 0, 11, 14, 9, 10, 12, 13, 3, 2, 5, 8, 4, 6, 7]

    files = [files[i] for i in file_idx]

    measurename = ['precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC',
                   'popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'PMI', 'IFA']
    mea_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    for i, j in zip(range(len(measurename)), mea_idx):
        results = []
        for file in files:
            file_path = os.path.join(path + '/', file)

            df = pd.read_csv(file_path, header=None)
            # results.append(df.iloc[:, j + 1])
            results.append(df.iloc[:, j])

        res = np.vstack(results)
        res = res.T
        data = pd.DataFrame(res)
        data.columns = methods
        data.to_csv(path + 'result_' + measurename[i] + '.csv', index=None, header=None)  # , header=None
