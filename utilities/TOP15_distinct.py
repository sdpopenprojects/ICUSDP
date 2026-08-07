import pandas as pd
import matplotlib.pyplot as plt
import os


# ==============================
# 输入文件
# ==============================

importance_file = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\visual\feature_analysis_results_top15\feature_importance_ranking_top15.csv"


save_dir = r"F:\ICUSDP\INTC\ICUSDP\W_C_test\visual\feature_analysis_results_top15\TOP15_distinct"


os.makedirs(save_dir, exist_ok=True)


save_fig = os.path.join(
    save_dir,
    "Feature_Category_Distribution.png"
)

# 增加 PDF 格式的保存路径
save_fig_pdf = os.path.join(
    save_dir,
    "Feature_Category_Distribution.pdf"
)

save_csv = os.path.join(
    save_dir,
    "Feature_Category_Distribution.csv"
)



# ==============================
# 读取Top15特征
# ==============================

df = pd.read_csv(importance_file)


# 按重要性排名排序，只取Top15
df = df.sort_values(
    by="排名"
).head(15)



print("\nTop15 Features:")
print(df)



# ==============================
# 软件度量类别定义
# ==============================

feature_category = {


    # ==================
    # Size Metrics
    # ==================

    "CountLineCode": "Size",
    "CountLine": "Size",
    "CountStmt": "Size",
    "CountStmtDecl": "Size",
    "CountLineCodeExe": "Size",
    "AvgLineCode": "Size",

    "CountDeclMethod": "Size",
    "CountDeclClass": "Size",



    # ==================
    # Complexity Metrics
    # ==================

    "SumCyclomatic": "Complexity",
    "SumCyclomaticStrict": "Complexity",
    "SumCyclomaticModified": "Complexity",

    "MaxCyclomatic": "Complexity",
    "MaxCyclomaticStrict": "Complexity",
    "MaxCyclomaticModified": "Complexity",

    "AvgCyclomatic": "Complexity",

    "CountPath_Max": "Complexity",
    "CountPath_Mean": "Complexity",

    "CountOutput_Max": "Complexity",

    "MaxNesting": "Complexity",



    # ==================
    # Change Metrics
    # ==================

    "MAJOR_COMMIT": "Change",
    "MINOR_COMMIT": "Change",
    "BUGFIX_COMMIT": "Change"

}



# ==============================
# 特征类别统计
# ==============================


category_count = {

    "Size":0,

    "Complexity":0,

    "Change":0

}


unknown=[]



for feature in df["特征"]:


    if feature in feature_category:

        category = feature_category[feature]

        category_count[category]+=1


    else:

        unknown.append(feature)



print("\nCategory Count:")
print(category_count)



if unknown:

    print("\n未分类特征:")
    print(unknown)



# ==============================
# 保存统计表
# ==============================


category_df = pd.DataFrame({

    "Category":list(category_count.keys()),

    "Number":[
        category_count[x]
        for x in category_count.keys()
    ]

})


# 百分比

total = category_df["Number"].sum()


category_df["Percentage"] = (
    category_df["Number"]
    /
    total
    *
    100
)



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
    figsize=(6,4)
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



# 添加百分比文字

for i,value in enumerate(category_df["Percentage"]):

    plt.text(

        i,

        value+1,

        f"{value:.1f}%",

        ha="center"

    )



plt.ylim(
    0,
    max(category_df["Percentage"])+15
)



plt.tight_layout()


# 1. 保存原有的 PNG 格式
plt.savefig(

    save_fig,

    dpi=600,

    bbox_inches="tight"

)

# 2. 多生成一下 PDF 格式的矢量图（指定 pdfTypeform1 确保科研论文引用时字体不失真）
plt.savefig(
    save_fig_pdf,
    bbox_inches="tight"
)


plt.show()



print("\n图片保存:")
print(save_fig)
print(save_fig_pdf)

print("\n表格保存:")
print(save_csv)