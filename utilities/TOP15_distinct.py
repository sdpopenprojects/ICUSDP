import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# 输入文件
# ==============================

importance_file = r"E:\ICUSDP-main\ICUSDP-main\visual\feature_analysis_results_top15\feature_importance_ranking_top15.csv"

save_dir = r"E:\ICUSDP-main\ICUSDP-main\visual\TOP15_distinct"

os.makedirs(save_dir, exist_ok=True)

save_fig = os.path.join(
    save_dir,
    "Feature_Category_Distribution.png"
)

# PDF 格式
save_fig_pdf = os.path.join(
    save_dir,
    "Feature_Category_Distribution.pdf"
)

# 统计结果 CSV
save_csv = os.path.join(
    save_dir,
    "Feature_Category_Distribution.csv"
)


# ==============================
# 读取 Top-15 特征
# ==============================

df = pd.read_csv(importance_file)

# 按重要性排名排序，只取 Top-15
df = df.sort_values(
    by="排名"
).head(15)

print("\nTop-15 Features:")
print(df)


# ============================================================
# 软件度量类别定义
#
# 一级分类：
#   1. Code Metrics
#      - Size
#      - Complexity
#
#   2. Process Metrics
#      - Change / Evolution
#
#   3. Ownership Metrics
#      - Developer / Ownership
# ============================================================

feature_category = {

    # ==============================
    # Code Metrics
    # ------------------------------
    # Size Metrics
    # ==============================

    "CountLineCode": "Code Metrics",
    "CountLine": "Code Metrics",
    "CountStmt": "Code Metrics",
    "CountStmtDecl": "Code Metrics",
    "CountLineCodeExe": "Code Metrics",
    "AvgLineCode": "Code Metrics",

    "CountDeclMethod": "Code Metrics",
    "CountDeclClass": "Code Metrics",


    # ==============================
    # Code Metrics
    # ------------------------------
    # Complexity Metrics
    # ==============================

    "SumCyclomatic": "Code Metrics",
    "SumCyclomaticStrict": "Code Metrics",
    "SumCyclomaticModified": "Code Metrics",

    "MaxCyclomatic": "Code Metrics",
    "MaxCyclomaticStrict": "Code Metrics",
    "MaxCyclomaticModified": "Code Metrics",

    "AvgCyclomatic": "Code Metrics",

    "CountPath_Max": "Code Metrics",
    "CountPath_Mean": "Code Metrics",

    "CountOutput_Max": "Code Metrics",

    "MaxNesting": "Code Metrics",




    "MAJOR_COMMIT": "Ownership Metrics",
    "MINOR_COMMIT": "Ownership Metrics",
    "BUGFIX_COMMIT": "Ownership Metrics",




}


# ==============================
# 特征类别统计
# ==============================

category_count = {
    "Code Metrics": 0,
    "Process Metrics": 0,
    "Ownership Metrics": 0
}

unknown = []


for feature in df["特征"]:

    if feature in feature_category:

        category = feature_category[feature]

        category_count[category] += 1

    else:

        unknown.append(feature)


# ==============================
# 输出分类结果
# ==============================

print("\nCategory Count:")
print(category_count)


if unknown:

    print("\n未分类特征:")
    print(unknown)


# ==============================
# 保存统计表
# ==============================

category_df = pd.DataFrame({

    "Category": list(category_count.keys()),

    "Number": [
        category_count[x]
        for x in category_count.keys()
    ]

})


# ==============================
# 计算百分比
# ==============================

total = category_df["Number"].sum()

if total > 0:

    category_df["Percentage"] = (
        category_df["Number"]
        / total
        * 100
    )

else:

    category_df["Percentage"] = 0


# 保存 CSV

category_df.to_csv(
    save_csv,
    index=False,
    encoding="utf-8-sig"
)


print("\n统计结果:")
print(category_df)


# ==============================
# 绘制柱状图
# ==============================

plt.figure(
    figsize=(6, 4)
)


plt.bar(
    category_df["Category"],
    category_df["Percentage"]
)


plt.xlabel(
    "Feature Category"
)

plt.ylabel(
    "Percentage (%)"
)


plt.title(
    "Top-15 Feature Category Distribution"
)


# ==============================
# 添加百分比文字
# ==============================

for i, value in enumerate(
    category_df["Percentage"]
):

    plt.text(
        i,
        value + 1,
        f"{value:.1f}%",
        ha="center"
    )


# ==============================
# 设置 Y 轴范围
# ==============================

max_percentage = category_df["Percentage"].max()

plt.ylim(
    0,
    max_percentage + 15
)


plt.tight_layout()


# ==============================
# 保存 PNG
# ==============================

plt.savefig(
    save_fig,
    dpi=600,
    bbox_inches="tight"
)


# ==============================
# 保存 PDF 矢量图
# ==============================

plt.savefig(
    save_fig_pdf,
    bbox_inches="tight"
)


plt.show()


# ==============================
# 输出保存路径
# ==============================

print("\n图片保存:")
print(save_fig)
print(save_fig_pdf)

print("\n表格保存:")
print(save_csv)