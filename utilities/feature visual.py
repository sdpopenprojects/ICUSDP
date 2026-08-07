import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 1. 核心配置（重点：替换真实特征名） --------------------------
# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 10)  # 适配纵向条形图的尺寸
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 文件路径
INPUT_FOLDER = Path(r"F:\ICUSDP\INTC\INTC\result_20251016\clustering\INTC_K-means")
OUTPUT_FOLDER = Path(r"F:\ICUSDP\INTC\INTC\result_20251016\visual\ICUSDP特征重要性图表")
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

# ====================== 重点：替换为你的65个真实特征名称 ======================
# 示例：从你的tree_rules中提取的特征名，你需要替换为实际的65个特征名
FEATURE_NAMES = [
    "CountLineCode", "CountPath_Max", "CountLineBlank", "COMM", "CountDeclMethodProtected",
    "CountDeclInstanceVariable", "CountLine", "MAJOR_COMMIT", "MaxCyclomaticModified",
    "CountLineCodeDecl", "CountOutput_Max", "CountDeclClassVariable", "Added_lines",
    "SumCyclomaticModified", "CountInput_Mean", "CountInput_Max", "CountOutput_Mean",
    "CountLineComment", "CyclomaticComplexity", "NPathComplexity",
    "CountDeclMethodPublic", "CountDeclMethodPrivate", "CountDeclClass", "CountDeclFunction",
    "CountLineCodeExec", "CountLineCodeExe", "CountLineCodeNonExec", "CountLinePreprocessor",
    "CountLineString", "CountLineChar", "CountLineNumber", "CountLineEmpty",
    "CountLineTodo", "CountLineDebug", "CountLineAssert", "CountLineMacro",
    "CountLineInclude", "CountLineDefine", "CountLineIf", "CountLineElse",
    "CountLineFor", "CountLineWhile", "CountLineDo", "CountLineSwitch",
    "CountLineCase", "CountLineBreak", "CountLineContinue", "CountLineGoto",
    "CountLineReturn", "CountLineThrow", "CountLineTry", "CountLineCatch",
    "CountLineFinally", "CountLineSynchronized", "CountLineVolatile", "CountLineConst",
    "CountLineStatic", "CountLineFinal", "CountLineAbstract", "CountLineInterface",
    "CountLineEnum", "CountLineAnnotation", "CountLinePackage", "CountLineImport",
    "CountLineClass"  # 共65个，按你的实际特征名替换
]
FEATURE_NUM = len(FEATURE_NAMES)  # 自动匹配特征数量


# -------------------------- 2. 读取特征重要性数据 --------------------------
def load_feature_importance_from_pkl():
    """读取PKL文件中的feature_importances，计算平均重要性"""
    feature_importance_list = []
    pkl_files = list(INPUT_FOLDER.glob("*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"在 {INPUT_FOLDER} 中未找到PKL文件！")

    print(f"🔍 找到 {len(pkl_files)} 个PKL文件，开始读取...")

    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            # 提取feature_importances（65行1列DataFrame）
            if "feature_importances" in data:
                feat_imp_df = data["feature_importances"]
                feat_imp = feat_imp_df.iloc[:, 0].values[:FEATURE_NUM]  # 提取第一列

                # 数据清洗
                feat_imp = np.nan_to_num(feat_imp, nan=0.0, posinf=0.0, neginf=0.0)
                feature_importance_list.append(feat_imp)
                print(f"✅ 成功读取 {pkl_file.name} | 特征数：{len(feat_imp)}")
            else:
                print(f"⚠️  {pkl_file.name} 无feature_importances Key，跳过")

        except Exception as e:
            print(f"❌ 读取 {pkl_file.name} 失败：{str(e)}")

    if not feature_importance_list:
        raise ValueError("未提取到任何有效特征重要性数据！")

    # 计算平均重要性 + 转换为百分比（归一化到0-100%）
    avg_feat_imp = np.mean(np.array(feature_importance_list), axis=0)
    avg_feat_imp_percent = (avg_feat_imp / avg_feat_imp.sum()) * 100  # 归一化为百分比
    print(f"\n📊 数据处理完成 | 有效文件数：{len(feature_importance_list)} | 特征总数：{len(avg_feat_imp_percent)}")
    print(f"📈 特征重要性总和（百分比）：{avg_feat_imp_percent.sum():.2f}%")
    return avg_feat_imp_percent, FEATURE_NAMES


# -------------------------- 3. 生成Top20特征重要性条形图（纵向+百分比） --------------------------
def plot_top20_feature_importance(avg_feat_imp_percent, feature_names):
    """
    生成Top20特征重要性条形图：
    - 纵坐标：真实特征名
    - 横坐标：百分比（%）
    - 按重要性降序排列
    """
    TOP_N = 20  # 展示Top20特征
    # 按重要性降序排序
    sorted_indices = np.argsort(avg_feat_imp_percent)[::-1]
    top_indices = sorted_indices[:TOP_N]

    # 提取Top20的特征名和百分比
    top_feat_names = [feature_names[i] for i in top_indices]
    top_feat_percent = avg_feat_imp_percent[top_indices]

    # 创建纵向条形图（barh）
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制条形图（前5个特征用强调色）
    colors = ["#E74C3C" if i < 5 else "#3498DB" for i in range(TOP_N)]
    bars = ax.barh(
        range(TOP_N),  # 纵坐标位置
        top_feat_percent,  # 横坐标值（百分比）
        color=colors,
        alpha=0.8,
        edgecolor='white',
        linewidth=1.2
    )

    # 设置纵坐标标签（真实特征名）
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(top_feat_names, fontsize=11)
    # 反转纵坐标（Top1在最上方）
    ax.invert_yaxis()

    # 设置横坐标标签（百分比）
    ax.set_xlabel("特征重要性（%）", fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title("ICUSDP 平均特征重要性 Top20（百分比）", fontsize=18, fontweight='bold', pad=20)

    # 添加数值标签（百分比格式，保留2位小数）
    for i, (bar, value) in enumerate(zip(bars, top_feat_percent)):
        ax.text(
            bar.get_width() + 0.1,  # 数值标签在条形右侧
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va='center',
            fontsize=9,
            fontweight='bold'
        )

    # 横坐标网格线
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
    # 调整横坐标范围（避免数值标签超出边界）
    ax.set_xlim(0, max(top_feat_percent) * 1.15)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    save_path = OUTPUT_FOLDER / "ICUSDP_Top20特征重要性_百分比_纵向图.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    # 生成Top20特征汇总表（CSV）
    feat_imp_df = pd.DataFrame({
        "排名": range(1, TOP_N + 1),
        "特征名称": top_feat_names,
        "重要性百分比(%)": top_feat_percent
    })
    csv_path = OUTPUT_FOLDER / "ICUSDP_Top20特征重要性汇总表.csv"
    feat_imp_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 输出结果
    print(f"\n✅ 纵向条形图已保存：{save_path}")
    print(f"✅ Top20汇总表已保存：{csv_path}")
    print("\n🏆 Top5重要特征（百分比）：")
    for i in range(5):
        print(f"   第{i + 1}名：{top_feat_names[i]} | {top_feat_percent[i]:.2f}%")


# -------------------------- 4. 主函数 --------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("🎯 开始生成ICUSDP Top20特征重要性（真实名称+百分比+纵向图）")
    print("=" * 80)

    try:
        # 读取数据（转换为百分比）
        avg_feat_imp_percent, feature_names = load_feature_importance_from_pkl()
        # 生成Top20纵向图
        plot_top20_feature_importance(avg_feat_imp_percent, feature_names)

        print("\n" + "=" * 80)
        print(f"🎉 所有文件已保存至：{OUTPUT_FOLDER}")
        print("=" * 80)
    except Exception as e:
        print(f"\n❌ 程序执行失败：{str(e)}")
        print("=" * 80)