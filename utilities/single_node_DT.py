import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pickle
import os
import glob
import numpy as np


def plot_decision_tree_sklearn(clf, feature_names, output_filename, figsize=(20, 10)):

    fig, ax = plt.subplots(figsize=figsize)


    if hasattr(clf, 'classes_'):
        class_names = [str(i) for i in clf.classes_]
    else:
        class_names = None


    tree_plot = plot_tree(clf,
                          feature_names=feature_names,
                          class_names=class_names,
                          label='all',
                          filled=True,
                          impurity=False,
                          node_ids=False,
                          proportion=False,
                          rounded=True,
                          fontsize=8,
                          ax=ax)


    for text in tree_plot:
        content = text.get_text()
        strs = content.split('\n')
        if len(strs) == 3:
            text.set_text(strs[-1])
        else:
            text.set_text(strs[0])


    base_name = os.path.splitext(os.path.basename(output_filename))[0]
    project_name = base_name.split('_decision_tree')[0]
    plt.title(f"Decision Tree - {project_name}", fontsize=16)

    plt.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  : {output_filename}")



def main():
    print("=" * 60)
    print("=" * 60)


    base_path = '../result_20251016/clustering/INTC_K-means/'


    if not os.path.exists(base_path):
        print(f"error - {base_path}")
        print(":", os.getcwd())
        return

    pkl_files = sorted(glob.glob(os.path.join(base_path, '*.pkl')))

    if not pkl_files:
        return

    print(f" {len(pkl_files)} ")


    output_dir = 'decision_trees_visualized'
    os.makedirs(output_dir, exist_ok=True)


    success_count = 0
    fail_count = 0


    for i, file_path in enumerate(pkl_files):
        file_name = os.path.basename(file_path)
        print(f"\n{'=' * 40}")
        print(f"处理文件 {i + 1}/{len(pkl_files)}: {file_name}")

        try:

            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            print(f" : {type(data)}")

            if not isinstance(data, dict):
                fail_count += 1
                continue


            if 'tree' not in data:
                fail_count += 1
                continue

            tree_model = data['tree']



            latent_dim = data.get('latent_dim', 65)
            feature_names = [f'latent_feature_{j}' for j in range(latent_dim)]
            print(f"  : {latent_dim}")


            base_name = os.path.splitext(file_name)[0]
            output_filename = os.path.join(output_dir, f"{base_name}_decision_tree.png")


            plot_decision_tree_sklearn(tree_model, feature_names, output_filename)


            try:
                if 'tree_max_depth' in data:
                    print(f"  : {data['tree_max_depth']}")
                if 'tree_n_nodes' in data:
                    print(f"  : {data['tree_n_nodes']}")


                if 'feature_importances' in data and data['feature_importances'] is not None:
                    importances = data['feature_importances']


                    if hasattr(importances, 'iloc'):

                        importances_array = importances.values.flatten()
                        print(f"  : pandas, : {importances.shape}")
                    elif isinstance(importances, (list, np.ndarray)):
                        importances_array = np.array(importances).flatten()
                        print(f"  : {type(importances).__name__}, : {len(importances_array)}")
                    else:
                        print(f"  : {type(importances)}")
                        importances_array = None

                    if importances_array is not None and len(importances_array) > 0:

                        non_zero_idx = np.where(importances_array > 0)[0]
                        if len(non_zero_idx) > 0:
                            top_n = min(3, len(non_zero_idx))

                            sorted_idx = np.argsort(importances_array[non_zero_idx])[-top_n:][::-1]
                            top_features = non_zero_idx[sorted_idx]
                            top_importances = importances_array[non_zero_idx][sorted_idx]

                            print(f"  {top_n}: {top_features}")
                            print(f"  : {top_importances}")
                        else:
                            print(f"  0")
            except Exception as info_error:
                print(f"  : {info_error}")

            success_count += 1

        except Exception as e:
            print(f"  ✗  {file_name} : {str(e)}")
            fail_count += 1

    # 汇总报告
    print(f"\n{'=' * 60}")
    print("over!")
    print(f"{'=' * 60}")
    print(f": {len(pkl_files)}")
    print(f": {success_count}")
    print(f": {fail_count}")
    print(f": {output_dir}")


    if success_count > 0:
        output_files = sorted(glob.glob(os.path.join(output_dir, '*.png')))
        for f in output_files:
            file_size = os.path.getsize(f) / 1024  # KB
            print(f"  {os.path.basename(f)} ({file_size:.1f} KB)")


if __name__ == "__main__":
    main()