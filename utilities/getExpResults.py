import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

if __name__ == '__main__':

    # path = r"F:\ICUSDP\INTC\ICUSDP\result_20260513\clustering"
    # path = r"F:\ICUSDP\INTC\ICUSDP\result_20260519\CAE_FS"
    path = r"E:\ICUSDP\INTC\ICUSDP-main\new\FS\disscussion\replace3"
    # path = r"F:\ICUSDP\INTC\ICUSDP\result_SDP\MD_FS"
    # path = r"F:\ICUSDP\INTC\ICUSDP\result_USDP\SC_FS"
    # path = r"F:\ICUSDP\INTC\ICUSDP\result_SDP\supervised_FS"


    # model_names = ['INTC_KMEANS_results']
    # model_names = ['ONE_Results']
    # model_names = ['INTC_MD_results']
    # model_names = ['INTC_SC_results']
    # model_names = ['DT_results', 'GBM_results', 'linearSVM_results', 'LR_results', 'RF_results', 'XGBoost_results']
    # model_names = ['ONE_results']
    model_names = ['INTC_KMEANS_GBM_results', 'INTC_KMEANS_RF_results', 'INTC_KMEANS_XGBOOST_results']


    for model_name in model_names:
        model_full_path = os.path.join(path, model_name)
        print(f": {model_full_path}")

        if not os.path.exists(model_full_path):
            continue

        files = [f for f in sorted(os.listdir(model_full_path)) if f.endswith('.csv')]
        if len(files) == 0:
            continue

        files_list = []
        results = []
        median_results = []

        for file in files:
            file_path = os.path.join(model_full_path, file)
            file_name = file[:-4]
            files_list.append(file_name)

            df = pd.read_csv(file_path, header=None)
            df = df.apply(pd.to_numeric, errors='coerce')
            results.append(df.values)


            res_raw = np.median(df.values, axis=0)


            m_time = np.median(df.iloc[:, -1])
            a_time = np.mean(df.iloc[:, -1])
            s_time = np.std(df.iloc[:, -1])


            res = np.hstack([res_raw[:31], m_time, a_time, s_time])
            median_results.append(res)


        results_df = DataFrame(np.vstack(results))
        results_df.to_csv(os.path.join(path, 'all_result_' + model_name + '.csv'), index=None, header=None)


        median_all_raw = np.median(results_df.values, axis=0)
        m_time_all = np.median(results_df.iloc[:, -1])
        a_time_all = np.mean(results_df.iloc[:, -1])
        std_time_all = np.std(results_df.iloc[:, -1])

        median_all = np.hstack([median_all_raw[:31], m_time_all, a_time_all, std_time_all])
        median_results.append(median_all)


        data = DataFrame(median_results)
        files_list.append('Median')
        data.index = files_list


        measurename = [
            'precision', 'recall', 'pf', 'F1', 'AUC', 'g_measure', 'g_mean', 'bal', 'MCC', 'accuracy',
            'Popt', 'cErecall', 'cEprecision', 'cEfmeasure', 'cMCC', 'cPMI', 'cIFA', 'cPCI', 'c_ROI_PII', 'c_ROI_PCI', 'ceIFA',
            'mRecall', 'mPrecision', 'mfmeasure', 'mMCC', 'mPMI', 'mIFA', 'mPCI', 'm_ROI_PII', 'm_ROI_PCI', 'meIFA',
            'median_time', 'mean_time', 'std_time'
        ]

        data.columns = measurename
        data.to_csv(os.path.join(path, 'result_' + model_name + '.csv'))

    print("\n over")