import os
import pandas as pd
from scipy.stats import spearmanr

folder = r"E:\ICUSDP-main\ICUSDP-main\visual\overlap15"

# ==========================
# ICUSDP
# ==========================
icusdp = pd.read_csv(
    os.path.join(folder, "ICUSDP_global_importance"),
    index_col=0
)

models = [
    "DT",
    "RF",
    "GBM",
    "XGBoost"
]

results = []

for model in models:

    other = pd.read_csv(
        os.path.join(folder, f"{model}_global_importance"),
        index_col=0
    )

    # 保证特征顺序一致
    common_features = icusdp.index.intersection(
        other.index
    )

    icusdp_rank = (
        icusdp.loc[common_features]
        .rank(ascending=False)
        .iloc[:, 0]
    )

    other_rank = (
        other.loc[common_features]
        .rank(ascending=False)
        .iloc[:, 0]
    )

    rho, pvalue = spearmanr(
        icusdp_rank,
        other_rank
    )

    results.append([
        model,
        round(rho, 4),
        pvalue
    ])

# ==========================
# 保存结果
# ==========================
result_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Spearman",
        "p-value"
    ]
)

print(result_df)

save_path = os.path.join(
    folder,
    "RQ4_Spearman.csv"
)

result_df.to_csv(
    save_path,
    index=False
)

print("\n保存成功:")
print(save_path)