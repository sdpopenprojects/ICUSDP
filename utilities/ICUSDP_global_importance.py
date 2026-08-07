import os
import pickle
import pandas as pd

# ==========================
# ICUSDP reports目录
# ==========================
#folder = r"F:\ICUSDP\INTC\ICUSDP\result_20260526_VAE\clustering\INTC_KMEANS\reports"
folder = r"F:\ICUSDP\INTC\ICUSDP\result_SDP\supervised\XGBoost\reports"

# ==========================
# 输出目录
# ==========================
save_dir = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\RQ3-5-feature"

os.makedirs(save_dir, exist_ok=True)

all_importances = []

# ==========================
# 读取28个项目
# ==========================
for file in os.listdir(folder):

    if not file.endswith(".pkl"):
        continue

    with open(os.path.join(folder, file), "rb") as f:
        report = pickle.load(f)

    imp_df = report['feature_importances']

    # 第一列就是重要性
    all_importances.append(imp_df.iloc[:, 0])

# ==========================
# 28项目平均
# ==========================
global_importance = pd.concat(
    all_importances,
    axis=1
).mean(axis=1)

global_importance = global_importance.sort_values(
    ascending=False
)

# ==========================
# 保存CSV
# ==========================
csv_path = os.path.join(
    save_dir,
    "XGBoost_global_importance.csv"
)

global_importance.to_csv(
    csv_path,
    header=["Importance"]
)

# ==========================
# 保存Top15
# ==========================
top15 = global_importance.head(15)

top15_path = os.path.join(
    save_dir,
    "XGBoost_Top15.csv"
)

top15.to_csv(
    top15_path,
    header=["Importance"]
)

# ==========================
# 打印结果
# ==========================
print("\n==============================")
print("ICUSDP Global Top15 Features")
print("==============================")
print(top15)

print("\n保存成功:")
print(csv_path)
print(top15_path)