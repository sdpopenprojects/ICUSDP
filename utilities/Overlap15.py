import os
import pandas as pd

folder = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\RQ3-5-feature"

# ==========================
# 读取ICUSDP Top15
# ==========================
icusdp = pd.read_csv(
    os.path.join(folder, "ICUSDP_Top15.csv"),
    index_col=0
)

icusdp_features = set(icusdp.index)

# ==========================
# 对比模型
# ==========================
models = [
    "DT",
    "RF",
    "GBM",
    "XGBoost"
]

results = []

for model in models:

    df = pd.read_csv(
        os.path.join(folder, f"{model}_Top15.csv"),
        index_col=0
    )

    model_features = set(df.index)

    overlap_num = len(
        icusdp_features.intersection(model_features)
    )

    overlap_rate = overlap_num / 15

    results.append([
        model,
        overlap_num,
        round(overlap_rate * 100, 2)
    ])

# ==========================
# 输出结果
# ==========================
result_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Overlap_Count",
        "Overlap@15(%)"
    ]
)

print(result_df)

# ==========================
# 保存
# ==========================
save_path = os.path.join(
    folder,
    "RQ3_Overlap15.csv"
)

result_df.to_csv(
    save_path,
    index=False
)

print("\n保存成功:")
print(save_path)