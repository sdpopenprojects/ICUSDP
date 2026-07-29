# normalized_feature_analysis_unified_top15.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def load_pkl_data(base_dir='../result_20251016/clustering/INTC_K-means/'):
    """
    加载所有pkl文件
    """
    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return None, None

    pkl_files = [f for f in os.listdir(base_dir) if f.endswith('.pkl')]
    pkl_files.sort()

    all_data_list = []
    all_feature_sets = []

    print(f"正在加载 {len(pkl_files)} 个pkl文件...")

    for i, filename in enumerate(pkl_files, 1):
        try:
            file_key = filename.replace('.pkl', '')
            file_path = os.path.join(base_dir, filename)

            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'feature_importances' in data:
                importances_df = data['feature_importances']

                if isinstance(importances_df, pd.DataFrame) and len(importances_df.columns) >= 1:
                    feature_names = importances_df.index.tolist()
                    importances = importances_df.iloc[:, 0].values

                    # 提取项目信息
                    if '_' in file_key and file_key.split('_')[-1].isdigit():
                        base_name = '_'.join(file_key.split('_')[:-1])
                        run_num = int(file_key.split('_')[-1])
                        display_name = f"{base_name} (运行{run_num})"
                    else:
                        base_name = file_key
                        run_num = 0
                        display_name = base_name

                    all_data_list.append({
                        'filename': filename,
                        'display_name': display_name,
                        'base_name': base_name,
                        'run_num': run_num,
                        'original_importances': importances,
                        'feature_names': feature_names
                    })

                    all_feature_sets.append(set(feature_names))
                    print(f"{i:2d}. ✓ {filename}")
                else:
                    print(f"{i:2d}. ✗ {filename}: 数据格式不符")
            else:
                print(f"{i:2d}. ✗ {filename}: 没有特征重要性数据")

        except Exception as e:
            print(f"{i:2d}. ✗ {filename}: 加载失败 - {e}")

    # 找出所有文件共有的特征
    if all_feature_sets:
        common_features = sorted(list(set.intersection(*all_feature_sets)))
        print(f"\n共有特征数量: {len(common_features)}")
        print(f"总文件数量: {len(all_data_list)}")
    else:
        common_features = []
        print("没有成功加载任何数据")

    return all_data_list, common_features


def normalize_importances(importances):
    """
    对单个文件的特征重要性进行归一化处理
    """
    if len(importances) == 0:
        return importances

    # 只考虑正的重要性值
    positive_importances = importances[importances > 0]

    if len(positive_importances) == 0:
        # 如果没有正的重要性值，返回零数组
        return np.zeros_like(importances)
    elif len(positive_importances) == 1:
        # 如果只有一个正的重要性值，将其设为1
        normalized = np.zeros_like(importances)
        normalized[importances > 0] = 1.0
        return normalized
    else:
        # Min-Max归一化
        min_val = np.min(positive_importances)
        max_val = np.max(positive_importances)

        if max_val > min_val:
            normalized = np.zeros_like(importances, dtype=float)
            for i, val in enumerate(importances):
                if val > 0:
                    normalized[i] = (val - min_val) / (max_val - min_val)
            return normalized
        else:
            # 如果所有正的重要性值相等，将它们设为1
            normalized = np.zeros_like(importances)
            normalized[importances > 0] = 1.0
            return normalized


def prepare_normalized_data(all_data_list, common_features):
    """
    准备归一化数据，确保所有分析使用相同的数据源
    """
    print(f"\n准备归一化数据...")

    n_files = len(all_data_list)
    n_features = len(common_features)

    # 创建特征名到索引的映射
    feature_to_idx = {feature: i for i, feature in enumerate(common_features)}

    # 初始化矩阵
    original_matrix = np.zeros((n_files, n_features))
    normalized_matrix = np.zeros((n_files, n_features))
    file_labels = []

    for i, record in enumerate(all_data_list):
        feature_names = record['feature_names']
        original_importances = record['original_importances']

        # 创建特征名到重要性的映射
        feature_dict = dict(zip(feature_names, original_importances))

        # 归一化处理
        normalized_vals = normalize_importances(original_importances)

        # 创建归一化特征映射
        normalized_dict = dict(zip(feature_names, normalized_vals))

        # 填充矩阵
        for j, feature in enumerate(common_features):
            if feature in feature_dict:
                original_matrix[i, j] = feature_dict[feature]
                normalized_matrix[i, j] = normalized_dict[feature]

        file_labels.append(record['display_name'])

    # 计算每个特征的平均重要性（考虑所有文件）
    original_means = np.mean(original_matrix, axis=0)
    normalized_means = np.mean(normalized_matrix, axis=0)

    return {
        'original_matrix': original_matrix,
        'normalized_matrix': normalized_matrix,
        'original_means': original_means,
        'normalized_means': normalized_means,
        'file_labels': file_labels,
        'common_features': common_features,
        'n_files': n_files,
        'n_features': n_features
    }


def create_normalized_bar_chart(data_dict, save_dir='./normalized_unified_top15/'):
    """
    创建归一化后的特征重要性条形图（Top 15）
    """
    os.makedirs(save_dir, exist_ok=True)

    normalized_means = data_dict['normalized_means']
    common_features = data_dict['common_features']
    n_files = data_dict['n_files']

    print(f"\n创建归一化特征重要性条形图 (Top 15)...")

    # 获取最重要的Top 15特征（按归一化重要性排序）
    top_n = 15
    sorted_indices = np.argsort(normalized_means)[::-1]
    top_indices = sorted_indices[:top_n]
    top_features = [common_features[i] for i in top_indices]
    top_normalized = normalized_means[top_indices]

    # 创建条形图
    plt.figure(figsize=(14, 8))

    # 使用viridis渐变色
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))

    # 创建水平条形图
    bars = plt.barh(range(top_n), top_normalized, color=colors, height=0.7)

    # 设置y轴标签（特征名称）
    plt.yticks(range(top_n), top_features, fontsize=11)
    plt.gca().invert_yaxis()  # 最重要的特征在顶部

    # 设置标题和标签
    plt.xlabel('平均归一化特征重要性 (0-1)', fontsize=12, fontweight='bold')
    plt.ylabel('特征名称', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} 特征平均归一化重要性 ({n_files} 个项目)',
              fontsize=14, fontweight='bold', pad=20)

    # 添加网格线
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')

    # 在条形右侧添加数值标签
    for bar, imp in zip(bars, top_normalized):
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{imp:.4f}', ha='left', va='center', fontsize=9)

    # 添加垂直参考线
    plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.text(0.51, top_n - 1, '中等重要性阈值', color='red', fontsize=9,
             verticalalignment='center')

    # 调整布局
    plt.tight_layout()

    # 保存图表
    save_path = os.path.join(save_dir, 'normalized_bar_chart_top15.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"归一化条形图 (Top 15) 已保存: {save_path}")

    # 保存数据
    df_ranking = pd.DataFrame({
        '特征': top_features,
        '平均归一化重要性': top_normalized,
        '排名': range(1, top_n + 1)
    })

    csv_path = os.path.join(save_dir, 'normalized_feature_ranking_top15.csv')
    df_ranking.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"归一化特征排名 (Top 15) 已保存: {csv_path}")

    return top_features, top_normalized, top_indices


def create_normalized_heatmap(data_dict, top_features, top_indices, save_dir='./normalized_unified_top15/'):
    """
    创建归一化后的特征重要性热力图 (Top 15)
    """
    print(f"\n创建归一化特征重要性热力图 (Top 15)...")

    normalized_matrix = data_dict['normalized_matrix']
    file_labels = data_dict['file_labels']
    n_files = data_dict['n_files']

    # 提取Top 15特征的归一化矩阵
    top_normalized_matrix = normalized_matrix[:, top_indices]

    # 创建热力图
    plt.figure(figsize=(16, max(8, n_files * 0.25)))

    im = plt.imshow(top_normalized_matrix, aspect='auto', cmap='YlOrRd',
                    interpolation='nearest', vmin=0, vmax=1)

    plt.xlabel('特征', fontsize=12, fontweight='bold')
    plt.ylabel('文件', fontsize=12, fontweight='bold')
    plt.title(f'归一化特征重要性热力图: {n_files} 个项目 × 15 个特征',
              fontsize=14, fontweight='bold', pad=20)

    plt.xticks(range(15), top_features, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(n_files), file_labels, fontsize=8)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('归一化重要性 (0-1)', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # 保存图表
    save_path = os.path.join(save_dir, 'normalized_heatmap_top15.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"归一化热力图 (Top 15) 已保存: {save_path}")

    # 保存热力图数据
    df_heatmap = pd.DataFrame(top_normalized_matrix,
                              index=file_labels,
                              columns=top_features)

    csv_path = os.path.join(save_dir, 'heatmap_data_top15.csv')
    df_heatmap.to_csv(csv_path, encoding='utf-8')

    print(f"热力图数据 (Top 15) 已保存: {csv_path}")


def create_normalization_comparison(data_dict, top_features, top_indices, save_dir='./normalized_unified_top15/'):
    """
    创建归一化前后的特征重要性对比图 (Top 15，按归一化重要性排序)
    """
    print(f"\n创建归一化前后对比图 (Top 15)...")

    original_means = data_dict['original_means']
    normalized_means = data_dict['normalized_means']
    n_files = data_dict['n_files']

    # 使用与条形图相同的Top 15特征
    top_original = original_means[top_indices]
    top_normalized = normalized_means[top_indices]

    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # 对比条形图
    x = np.arange(15)
    width = 0.35

    # 原始值条形图
    bars1 = ax1.bar(x - width / 2, top_original, width,
                    label='原始重要性', color='steelblue', alpha=0.8)
    # 归一化值条形图
    bars2 = ax1.bar(x + width / 2, top_normalized, width,
                    label='归一化重要性', color='lightcoral', alpha=0.8)

    ax1.set_xlabel('特征', fontsize=11, fontweight='bold')
    ax1.set_ylabel('重要性值', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top 15 特征：归一化前后对比 (按归一化重要性排序)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_features, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # 只显示大于0的值
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 计算归一化效果（百分比变化）
    normalization_effects = []
    for i in range(15):
        orig_val = top_original[i]
        norm_val = top_normalized[i]

        if orig_val > 0:
            # 计算归一化后的相对变化
            change = ((norm_val / orig_val) if orig_val > 0 else 0) - 1
            normalization_effects.append(change * 100)  # 百分比变化
        else:
            normalization_effects.append(0)

    # 归一化效果条形图
    colors_effect = ['green' if eff >= 0 else 'red' for eff in normalization_effects]
    bars_effect = ax2.bar(x, normalization_effects, color=colors_effect, alpha=0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('特征', fontsize=11, fontweight='bold')
    ax2.set_ylabel('归一化效果 (%)', fontsize=11, fontweight='bold')
    ax2.set_title('归一化对特征排名的影响（正值为提升，负值为下降）', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_features, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 添加效果数值标签
    for bar, effect in zip(bars_effect, normalization_effects):
        height = bar.get_height()
        if abs(height) > 0.1:  # 只显示变化明显的
            ax2.text(bar.get_x() + bar.get_width() / 2.,
                     height + (0.5 if height >= 0 else -1),
                     f'{effect:.1f}%',
                     ha='center', va='center' if height >= 0 else 'top',
                     fontsize=8, fontweight='bold')

    plt.tight_layout()

    # 保存对比图
    save_path = os.path.join(save_dir, 'normalization_comparison_top15.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"归一化对比图 (Top 15) 已保存: {save_path}")

    # 保存对比数据
    df_comparison = pd.DataFrame({
        '特征': top_features,
        '原始平均重要性': top_original,
        '归一化平均重要性': top_normalized,
        '变化百分比(%)': normalization_effects,
        '归一化排名': range(1, 16)  # 归一化排序
    })

    # 计算原始排名
    all_indices = np.argsort(data_dict['original_means'])[::-1]
    original_ranks = {}
    for rank, idx in enumerate(all_indices):
        original_ranks[idx] = rank + 1

    df_comparison['原始排名'] = [original_ranks[idx] for idx in top_indices]
    df_comparison['排名变化'] = df_comparison['原始排名'] - df_comparison['归一化排名']

    csv_path = os.path.join(save_dir, 'normalization_comparison_data_top15.csv')
    df_comparison.to_csv(csv_path, index=False, encoding='utf-8')

    print(f"归一化对比数据 (Top 15) 已保存: {csv_path}")

    # 打印排名变化
    print("\n" + "=" * 80)
    print("特征排名变化 (负值表示归一化后排名提升):")
    print("=" * 80)
    for _, row in df_comparison.iterrows():
        change_str = f"下降{row['排名变化']}位" if row['排名变化'] > 0 else f"提升{abs(row['排名变化'])}位"
        print(f"{row['特征']:30s} | 原始排名: {row['原始排名']:2d} → 归一化排名: {row['归一化排名']:2d} ({change_str})")


def create_summary_report(data_dict, top_features, top_indices, save_dir='./normalized_unified_top15/'):
    """
    创建分析摘要报告
    """
    print(f"\n生成分析摘要...")

    n_files = data_dict['n_files']
    original_means = data_dict['original_means']
    normalized_means = data_dict['normalized_means']

    # 提取Top 15特征的统计
    top_original = original_means[top_indices]
    top_normalized = normalized_means[top_indices]

    # 创建统计摘要
    stats_data = []
    for i, feature in enumerate(top_features):
        stats_data.append({
            '特征': feature,
            '归一化平均重要性': top_normalized[i],
            '原始平均重要性': top_original[i],
            '归一化排名': i + 1
        })

    df_stats = pd.DataFrame(stats_data)

    # 保存统计摘要
    stats_path = os.path.join(save_dir, 'feature_statistics_summary_top15.csv')
    df_stats.to_csv(stats_path, index=False, encoding='utf-8')

    print(f"特征统计摘要 (Top 15) 已保存: {stats_path}")

    return df_stats


def main():
    """
    主函数
    """
    print("开始归一化特征重要性分析 (统一Top 15)...")
    print("=" * 70)

    # 设置matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150

    # 1. 加载数据
    base_dir = '../result_20251016/clustering/INTC_K-means/'
    all_data_list, common_features = load_pkl_data(base_dir)

    if not all_data_list:
        print("没有找到可用的数据，程序退出")
        return

    print(f"\n数据加载完成:")
    print(f"- 文件数量: {len(all_data_list)}")
    print(f"- 共有特征数量: {len(common_features)}")

    # 2. 准备归一化数据（确保数据一致性）
    data_dict = prepare_normalized_data(all_data_list, common_features)

    # 创建输出目录
    output_dir = './normalized_unified_top15/'
    os.makedirs(output_dir, exist_ok=True)

    # 3. 创建归一化条形图 (Top 15)
    top_features, top_normalized, top_indices = create_normalized_bar_chart(data_dict, output_dir)

    # 4. 创建归一化热力图 (Top 15)
    create_normalized_heatmap(data_dict, top_features, top_indices, output_dir)

    # 5. 创建归一化前后对比图 (Top 15)
    create_normalization_comparison(data_dict, top_features, top_indices, output_dir)

    # 6. 创建分析摘要
    df_stats = create_summary_report(data_dict, top_features, top_indices, output_dir)

    print("\n" + "=" * 70)
    print("分析完成!")
    print(f"所有输出文件保存在: {output_dir}")
    print("=" * 70)

    # 打印最重要的5个特征
    print("\nTop 5 最重要的特征 (归一化后):")
    print("-" * 60)
    for i in range(min(5, len(top_features))):
        print(f"{i + 1}. {top_features[i]:35s}: 归一化重要性 = {top_normalized[i]:.4f}")


if __name__ == '__main__':
    main()