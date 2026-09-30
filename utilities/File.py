import os
import pandas as pd
import pickle

def create_dir(dirname):
    # path = os.getcwd() + '/' + dirname
    path = dirname
    folder = os.path.exists(path)

    try:
        if not folder:
            os.makedirs(path, exist_ok=True)
    except OSError as err:
        print(err)

    return path + "/"


def save_results(save_path, score):
    # with open(fres + project_name + '-' + model_name + '.csv', 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(score)

    # pandas
    tempRes = pd.DataFrame(score).T
    tempRes.to_csv(save_path + '.csv', index=False, header=False, mode='a')


def save_report_pickle(report, filename="interpretability_report.pkl"):
    """ save Pickle """
    with open(filename, 'ab') as f:
        pickle.dump(report, f)

def save_results_pickle(save_path, results):
    with open(save_path + '.pkl', 'ab') as f:
        pickle.dump(results, f)
    f.close()


def load_results_pickle(save_path):
    # with open(save_path + '.pkl', 'rb') as f:
    with open(save_path, 'rb') as f:
        results = pickle.load(f)

    f.close()

    return results

def load_results_pickle_v2(save_path):
    pkl_data = []
    with open(save_path, 'rb') as f:
        result = pickle.load(f)
        pkl_data.append(result)

        while True:
            try:
                result = pickle.load(f)
                pkl_data.append(result)
            except EOFError:
                break
    f.close()

    return pkl_data