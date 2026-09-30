import os
import pickle
import pandas as pd

base_dir = r"E:\ICUSDP\INTC\ICUSDP-main\new\FS\RQ4"
algorithms = ["DT", "GBM", "ICUSDP", "linearSVM", "LR", "RF", "XGBoost"]



class DummyClass:

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class SafeUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if "sklearn" in module or name in ["Tree", "TreeBuilder"]:
            return DummyClass
        return super().find_class(module, name)



output_dir = os.path.join(base_dir, "pkl_results")
os.makedirs(output_dir, exist_ok=True)


for algo in algorithms:
    algo_dir = os.path.join(base_dir, algo)

    if not os.path.exists(algo_dir):
        print(f" {algo_dir}，")
        continue

    algo_proj_dict = {}


    for root, dirs, files in os.walk(algo_dir):
        pkl_files = sorted([f for f in files if f.endswith(".pkl")])

        for file in pkl_files:
            file_path = os.path.join(root, file)
            project_name = os.path.splitext(file)[0]

            try:
                with open(file_path, "rb") as f:
                    data = SafeUnpickler(f).load()

                imp_dict = {}
                if isinstance(data, dict) and "feature_importances" in data:
                    df_imp = data["feature_importances"]

                    if isinstance(df_imp, pd.DataFrame):
                        if (
                            "feature" in df_imp.columns
                            and "importance" in df_imp.columns
                        ):
                            imp_dict = dict(
                                zip(df_imp["feature"], df_imp["importance"])
                            )
                        else:
                            imp_dict = df_imp.iloc[:, 0].to_dict()
                    elif isinstance(df_imp, pd.Series):
                        imp_dict = df_imp.to_dict()
                    elif isinstance(df_imp, dict):
                        imp_dict = df_imp

                if imp_dict:

                    algo_proj_dict[project_name] = imp_dict
                else:
                    print(
                        f" feature_importances "
                    )

            except Exception as e:
                print(f"read {file_path} fail: {e}")


    if algo_proj_dict:

        df_algo = pd.DataFrame.from_dict(algo_proj_dict, orient="index").fillna(
            0
        )


        combined_csv_path = os.path.join(
            output_dir, f"skesd_feat_imp_{algo}.csv"
        )
        df_algo.to_csv(combined_csv_path, index_label="Project")

        print(
            f"success [{algo}]  CSV ( {len(df_algo)} )，: {combined_csv_path}"
        )

print("\n run over。")