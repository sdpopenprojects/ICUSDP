import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

if __name__ == '__main__':

    cols = [i for i in range(31)]
    feas = pd.read_csv(os.path.abspath('../result/pub_fs_iter10/selected_features.csv'), names=cols)

    # feas = pd.read_csv(os.path.abspath("../JIRA28/activemq-5.0.0.csv"))
    # featList = feas.columns.values
    # featList = featList[:-1]

    path = r"/home/ubuntu/myResearch/experiments/MUSDPnew/result/pub_fs_iter10/"
    model_name = 'MUSDP_perm_permfeaimp2'

    files = [f for f in sorted(os.listdir(path + model_name + '/')) if f.endswith('.csv')]
    files_num = len(files)

    files_list = []
    results = []
    average_results = []

    for i in range(len(files)):
        file = files[i]
        file_path = os.path.join(path + model_name + '/', file)
        file_name = file[:-4]
        files_list.append(file_name)

        df = pd.read_csv(file_path, header=None)
        # results.append(np.array(df))

        df.columns = feas.iloc[i].dropna()
        df.to_csv(path + model_name +'/' + 'feature' + '/' + file_name+ '.csv', index=False, header=True)

        # res = np.median(df, axis=0)
        res = np.mean(df, axis=0)
        average_results.append(res)


    # for file in files:
    #     file_path = os.path.join(path + model_name + '/', file)
    #     file_name = file[:-4]
    #     files_list.append(file_name)
    #
    #     df = pd.read_csv(file_path, header=None)
    #     # results.append(np.array(df))
    #
    #     # res = np.median(df, axis=0)
    #     res = np.mean(df, axis=0)
    #     average_results.append(res)
    #
    # # save to csv file
    # results = DataFrame(np.vstack(results))
    # results.to_csv(path + 'all_result_' + model_name + '.csv', index=None, header=None)
    #
    # median_all = np.median(results, axis=0)
    # average_results.append(median_all)
    # mean_all = np.mean(results, axis=0)
    # average_results.append(mean_all)
    #
    # data = DataFrame(average_results)
    #
    # # files_list.append('Mean')
    # # data.index = files_list
    # # data.columns = featList
    #
    # # select the columns that the mean values are larger than 0
    # # mean_res = data.mean(axis=0)
    # # median_res = data.median(axis=0)
    # # data = pd.concat([data, pd.DataFrame(mean_res).T], axis=0, ignore_index=True)
    # # data = pd.concat([data, pd.DataFrame(median_res).T], axis=0, ignore_index=True)
    #
    # files_list.append('Median')
    # files_list.append('Mean')
    # data.index = files_list
    # data = data.sort_values(by=['Mean', 'Median'], axis=1, ascending=False)
    #
    # data.to_csv(path + 'permFeaImp_' + model_name + '.csv')
