import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ================== 1. 配置路径 ==================
data_dir = r"E:\ICUSDP-main\ICUSDP-main\visual\overlap15"

# ================== 2. 读取所有 CSV 文件 ==================
feature_lists = {}
file_pattern = os.path.join(data_dir, "*_global_importance.csv")

for file_path in glob.glob(file_pattern):
    file_name = os.path.basename(file_path)
    model_name = file_name.replace("_global_importance.csv", "")
    df = pd.read_csv(file_path)

    if 'Feature' in df.columns and 'Importance' in df.columns:
        sorted_features = df.sort_values(by="Importance", ascending=False)["Feature"].tolist()
    else:
        sorted_features = df.sort_values(by=df.columns[1], ascending=False).iloc[:, 0].tolist()

    feature_lists[model_name] = sorted_features
    print(f"✅ 已读取 {model_name}: {len(sorted_features)} 个特征")

# ================== 3. 基础校验 ==================
if "ICUSDP" not in feature_lists:
    raise ValueError("❌ 错误: 必须包含 ICUSDP_global_importance.csv！")

# ================== 4. 关键修改点 ==================
# ★★★ 强制设定总特征池为 65（因为论文里明确说了原始指标一共 65 个）★★★
TOTAL_FEATURES = 65

# K 的取值范围：如果你的文件只有 15 个特征，max_k 就取 15；否则取 65
max_k = min(len(feature_lists["ICUSDP"]), TOTAL_FEATURES)
K_values = list(range(5, max_k + 1, 5))
if max_k not in K_values:
    K_values.append(max_k)
K_values = sorted(set(K_values))


def compute_overlap(list_a, list_b, k):
    return len(set(list_a[:k]) & set(list_b[:k])) / k * 100


# ================== 5. 计算正确的递增随机基线 ==================
# ★★★ 正确公式：K / 65 * 100% ★★★
random_baseline = [k / TOTAL_FEATURES * 100 for k in K_values]

# ★★★ 打印出来校验一下，看看是不是 [7.69, 15.38, 23.08, 30.77, 38.46, 46.15, 53.85, 61.54, 69.23, 76.92, 84.62, 92.31, 100.0] ★★★
print("\n📊 随机基线校验 (Random Expectation):")
for k, base in zip(K_values, random_baseline):
    print(f"   K={k:2d} -> {base:.2f}%")

# ================== 6. 绘图 ==================
plt.figure(figsize=(10, 6))

for model in ["RF", "GBM", "XGBoost", "DT"]:
    if model not in feature_lists:
        continue
    overlaps = [compute_overlap(feature_lists["ICUSDP"], feature_lists[model], k) for k in K_values]
    plt.plot(K_values, overlaps, marker='o', linewidth=2, label=f"ICUSDP vs {model}")

# 画基线
plt.plot(K_values, random_baseline, 'k--', linewidth=2, label=f"Random Expectation (K/65)")

# 图表装饰
plt.xlabel("Top-K Features", fontsize=18)
plt.ylabel("Overlap Rate (%)", fontsize=18)
plt.title("Top-K Feature Overlap between ICUSDP and Supervised Models", fontsize=14)
plt.legend(loc="best", fontsize=14)
plt.grid(alpha=0.3)
# plt.xticks(K_values)
plt.xticks(K_values, fontsize=11)  # X轴刻度字体大小
plt.yticks(fontsize=11)            # Y轴刻度字体大小
plt.tight_layout()

# 保存
plt.savefig(os.path.join(data_dir, "overlap_curve.pdf"), dpi=300)
plt.show()

print(f"\n✅ 修正后的图片已保存为 overlap_curve.pdf")