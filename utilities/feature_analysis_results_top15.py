# merge_all_features_one_analysis_top15.py
import os
import sys
import numpy as np

# ================= 强行兼容旧版本 pkl 的补丁 =================
try:
    import sklearn.tree._tree as sklearn_tree

    # 检查当前 sklearn 是否是带新字段的高版本
    if hasattr(sklearn_tree, 'NODE_DTYPE') and 'missing_go_to_left' in sklearn_tree.NODE_DTYPE.names:
        # 备份原本的复合类型
        old_dtype = sklearn_tree.NODE_DTYPE

        # 构建一个不带 'missing_go_to_left' 的旧版本数据结构
        formats = ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8']
        names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples',
                 'weighted_n_node_samples']

        # 强制覆盖 sklearn 内部的 NODE_DTYPE，让它以旧结构的视角去反序列化
        sklearn_tree.NODE_DTYPE = np.dtype({'names': names, 'formats': formats})
        print("[INFO] 已成功应用旧版决策树 pkl 兼容性补丁。")
except Exception as e:
    print(f"[WARN] 补丁应用失败 (可能无需补丁): {e}")
# =============================================================

import pickle
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def load_pkl_files(base_dir='E:/ICUSDP-main/ICUSDP-main/result_ICUSDP/INTC_KMEANS/reports/'):
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

                    # 【修正细节 1】：完美保留原文件名，去除带有小括号的错误解析，将下划线替换成点或直接保留
                    # 直接使用 file_key 作为 display_name，但把 groovy 的下划线保留原样，避免拆出括号
                    display_name = file_key.replace('_BETA_', '-BETA-')  # 保护 BETA 字段
                    if '_' in display_name and display_name.split('_')[-1].isdigit() and '-' not in \
                            display_name.split('_')[-1]:
                        # 如果是真正的类似 _1 结尾但不是 groovy 的复合结构，可以做处理。
                        # 为稳妥起见，直接让 display_name 等于原 file_key，只做格式美化，彻底移除小括号逻辑。
                        display_name = file_key.replace('_', '.')
                    else:
                        display_name = file_key

                    all_data_list.append({
                        'filename': filename,
                        'display_name': display_name,
                        'base_name': file_key,
                        'run_num': 0,
                        'importances': importances,
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

    if all_feature_sets:
        common_features = sorted(list(set.intersection(*all_feature_sets)))
        print(f"\n共有特征数量: {len(common_features)}")
        print(f"总项目数量: {len(all_data_list)}")
    else:
        common_features = []
        print("没有成功加载任何数据")

    return all_data_list, common_features


def create_feature_importance_bar_chart(all_data_list, common_features, save_dir='./analysis_results/'):
    """
    创建特征重要性条形图（单独图表）- 修改为Top 15
    """
    os.makedirs(save_dir, exist_ok=True)

    n_files = len(all_data_list)
    n_features = len(common_features)

    print(f"\n创建特征重要性条形图...")

    importance_matrix = np.zeros((n_files, n_features))
    file_labels = []

    feature_to_idx = {feature: i for i, feature in enumerate(common_features)}

    for i, record in enumerate(all_data_list):
        feature_names = record['feature_names']
        importances = record['importances']
        file_labels.append(record['display_name'])


        feature_dict = dict(zip(feature_names, importances))

        for j, feature in enumerate(common_features):
            if feature in feature_dict:
                importance_matrix[i, j] = feature_dict[feature]

    avg_importances = np.mean(importance_matrix, axis=0)

    top_n = 15
    sorted_indices = np.argsort(avg_importances)[::-1]
    top_indices = sorted_indices[:top_n]
    top_features = [common_features[i] for i in top_indices]
    top_avg_importances = avg_importances[top_indices]

    plt.figure(figsize=(14, 10))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))

    bars = plt.barh(range(top_n), top_avg_importances, color=colors, height=0.7)

    plt.yticks(range(top_n), top_features, fontsize=22, fontweight='bold')
    plt.gca().invert_yaxis()

    # 【修正细节 2】：动态拓宽 X 轴的右边界，防止最大特征的数值文本戳破或者越过右边框线
    max_val = np.max(top_avg_importances)
    plt.xlim(0, max_val * 1.18)

    plt.xticks(fontsize=18)
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')

    for bar, imp in zip(bars, top_avg_importances):
        width = bar.get_width()
        plt.text(width + (max_val * 0.01), bar.get_y() + bar.get_height() / 2,
                 f'{imp:.4f}', ha='left', va='center', fontsize=18, fontweight='bold')

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'feature_importance_bar_chart_top15.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    save_path_pdf = os.path.join(save_dir, 'feature_importance_bar_chart_top15.pdf')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')

    plt.close()
    print(f"条形图已保存: {save_path} 及其 PDF 格式")

    df_avg_importance = pd.DataFrame({
        '特征': top_features,
        '平均重要性': top_avg_importances,
        '排名': range(1, top_n + 1)
    })
    csv_path = os.path.join(save_dir, 'feature_importance_ranking_top15.csv')
    df_avg_importance.to_csv(csv_path, index=False, encoding='utf-8')

    return top_features, top_avg_importances, importance_matrix


def create_heatmap_visualization(all_data_list, common_features, top_features, save_dir='./analysis_results/'):
    """
    创建热力图（单独图表） - 修改为Top 15
    """
    print(f"\n创建热力图...")

    n_files = len(all_data_list)
    top_n = len(top_features)

    feature_to_idx = {feature: i for i, feature in enumerate(common_features)}
    top_indices = [feature_to_idx[f] for f in top_features]

    importance_matrix = np.zeros((n_files, top_n))
    file_labels = []

    # for i, record in enumerate(all_data_list):
    #     feature_names = record['feature_names']
    #     importances = record['importances']
    #     file_labels.append(record['display_name'])
    #
    #     feature_dict = dict(zip(feature_names, importances))

    for i, record in enumerate(all_data_list):
        feature_names = record['feature_names']
        importances = record['importances']

        # 仅用于热力图纵轴显示的项目名称简化
        project_name = record['base_name']

        if project_name == 'groovy-1_6_BETA_1':
            project_name = 'groovy-1_6_beta1'

        elif project_name == 'groovy-1_6_BETA_2':
            project_name = 'groovy-1_6_beta2'

        elif project_name == 'jruby-1.7.0.preview1':
            project_name = 'jruby-1.7.0'

        elif project_name == 'wicket-1.3.0-incubating-beta-1':
            project_name = 'wicket-1.3.0-beta1'

        file_labels.append(project_name)

        feature_dict = dict(zip(feature_names, importances))



        for j, feature in enumerate(top_features):
            if feature in feature_dict:
                importance_matrix[i, j] = feature_dict[feature]

    norm_matrix = importance_matrix.copy()
    if np.max(norm_matrix) > 0:
        norm_matrix = norm_matrix / np.max(norm_matrix)

    plt.figure(figsize=(16, max(10, n_files * 0.35)))

    cmap = plt.cm.YlOrRd

    im = plt.imshow(norm_matrix, aspect='auto', cmap=cmap,
                    interpolation='nearest', vmin=0, vmax=1)

    # 💡 核心修改点：去掉了横纵轴标签以及 colorbar 标签的 fontweight='bold'，但完全保留了 20, 18, 22 等大字号
    plt.xticks(range(top_n), top_features, rotation=45, ha='right', fontsize=24)
    plt.yticks(range(n_files), file_labels, fontsize=22)

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Normalization Importance', fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    plt.grid(False)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'feature_importance_heatmap_top15.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    save_path_pdf = os.path.join(save_dir, 'feature_importance_heatmap_top15.pdf')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')

    plt.close()
    print(f"热力图已保存: {save_path} 及其 PDF 格式")

    df_heatmap = pd.DataFrame(norm_matrix, index=file_labels, columns=top_features)
    csv_path = os.path.join(save_dir, 'heatmap_data_top15.csv')
    df_heatmap.to_csv(csv_path, encoding='utf-8')

    return norm_matrix


def create_comprehensive_statistics(all_data_list, common_features, save_dir='./analysis_results/'):
    """
    创建综合统计摘要
    """
    print(f"\n生成综合统计摘要...")

    n_files = len(all_data_list)
    n_features = len(common_features)

    importance_matrix = np.zeros((n_files, n_features))

    for i, record in enumerate(all_data_list):
        feature_names = record['feature_names']
        importances = record['importances']
        feature_dict = dict(zip(feature_names, importances))

        for j, feature in enumerate(common_features):
            if feature in feature_dict:
                importance_matrix[i, j] = feature_dict[feature]

    stats_data = []

    for j, feature in enumerate(common_features):
        feature_importances = importance_matrix[:, j]
        non_zero = feature_importances[feature_importances > 0]

        if len(non_zero) > 0:
            stats_data.append({
                '特征': feature,
                '出现文件数': len(non_zero),
                '出现比例': len(non_zero) / n_files,
                '平均重要性': np.mean(non_zero),
                '中位数重要性': np.median(non_zero),
                '重要性标准差': np.std(non_zero),
                '最大重要性': np.max(non_zero),
                '最小重要性': np.min(non_zero),
                '重要性总和': np.sum(non_zero)
            })

    df_stats = pd.DataFrame(stats_data)
    df_stats = df_stats.sort_values('平均重要性', ascending=False)

    stats_path = os.path.join(save_dir, 'comprehensive_statistics.csv')
    df_stats.to_csv(stats_path, index=False, encoding='utf-8')
    print(f"综合统计数据已保存: {stats_path}")

    return df_stats


def main():
    print("开始分析所有pkl文件...")
    print("=" * 70)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150

    base_dir = 'E:/ICUSDP-main/ICUSDP-main/result_ICUSDP/INTC_KMEANS/reports/'
    all_data_list, common_features = load_pkl_files(base_dir)

    if not all_data_list:
        print("没有找到可用的数据，程序退出")
        return

    output_dir = 'E:/ICUSDP-main/ICUSDP-main/visual/feature_analysis_results_top15/'
    os.makedirs(output_dir, exist_ok=True)

    top_features, top_importances, importance_matrix = create_feature_importance_bar_chart(
        all_data_list, common_features, output_dir
    )

    create_heatmap_visualization(
        all_data_list, common_features, top_features, output_dir
    )

    df_stats = create_comprehensive_statistics(
        all_data_list, common_features, output_dir
    )

    print("\n" + "=" * 70)
    print("Analysis Finished Successfully! All modifications applied.")
    print("=" * 70)


if __name__ == '__main__':
    main()