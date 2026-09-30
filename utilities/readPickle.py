import os

from utilities.File import load_results_pickle_v2

# path = r"../result_20250930/"
# model_name = 'ICluVAE_130-65-65_iter20_random_state'
path = '../result_20251016/clustering/'
model_name = ('INTC_K-means')

files = [f for f in sorted(os.listdir(path + model_name + '/')) if f.endswith('.pkl')]
files_num = len(files)

files_list = []
for file in files:
    file_path = os.path.join(path + model_name + '/', file)
    file_name = file[:-4]
    files_list.append(file_name)

    reports = load_results_pickle_v2(file_path)
    for i in range(len(reports)):
        report = reports[i]
        tree_max_depth = report['tree_max_depth']
        tree_n_nodes = report['tree_n_nodes']
        # view_importances = report['view_importances']
        feature_importances = report['feature_importances']
        pseudo_labels = report['pseudo_labels']
        latent_dim = report['latent_dim']
        # n_vaes = report['n_vaes']
        n_vaes = report.get('n_vaes', 'N/A')



    print(file_name + ' done !')



