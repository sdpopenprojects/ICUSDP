# visualize_from_saved_data_simple.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.tree import plot_tree, export_text
import matplotlib.patches as mpatches


def load_report_data(project_name, model_name='INTC_K-means'):
    """
    从保存的pickle文件中加载报告数据
    """
    report_dir = '../result_20251016/clustering/'
    report_file = os.path.join(report_dir, model_name, project_name)

    if not os.path.exists(report_file):
        if not report_file.endswith('.pkl'):
            report_file = report_file + '.pkl'
        if not os.path.exists(report_file):
            report_file_with_number = report_file.replace('.pkl', '_0.pkl')
            if os.path.exists(report_file_with_number):
                report_file = report_file_with_number
            else:
                raise FileNotFoundError(f"找不到报告文件: {report_file}")

    with open(report_file, 'rb') as f:
        report_data = pickle.load(f)

    print(f"成功加载项目 '{project_name}' 的报告数据")
    return report_data


def extract_feature_info(report_data):
    """
    从报告数据中提取特征信息
    """
    decision_tree = None
    if 'tree' in report_data:
        decision_tree = report_data['tree']
        print("从报告中获取决策树对象")

    importances_df = None
    if 'feature_importances' in report_data and isinstance(report_data['feature_importances'], pd.DataFrame):
        importances_df = report_data['feature_importances']
        print("从报告中获取特征重要性DataFrame")

    feature_names = None
    importances = None

    if importances_df is not None:
        feature_names = importances_df.index.tolist()
        if len(importances_df.columns) == 1:
            importances = importances_df.iloc[:, 0].values
        else:
            importances = importances_df.iloc[:, 0].values
            print(f"警告：特征重要性DataFrame有 {len(importances_df.columns)} 列，使用第一列")

    if importances is None and decision_tree is not None:
        if hasattr(decision_tree, 'feature_importances_'):
            importances = decision_tree.feature_importances_
            print("从决策树对象获取特征重要性")

    if feature_names is None and importances is not None:
        n_features = len(importances)
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        print(f"创建默认特征名称，共 {n_features} 个特征")

    return decision_tree, feature_names, importances, importances_df


def visualize_decision_tree_minimal(decision_tree, feature_names, project_name, max_depth=5):
    """
    最小化决策树可视化 - 每个节点只显示一个信息

    Args:
        decision_tree: 决策树对象
        feature_names: 特征名称列表
        project_name: 项目名称
        max_depth: 最大显示深度，默认为5
    """
    if decision_tree is None:
        print("警告：没有可用的决策树对象")
        return None

    # 创建输出目录
    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # 准备特征名称列表（使用简短名称）
    if feature_names is not None:
        tree_feature_names = feature_names
    else:
        if hasattr(decision_tree, 'n_features_in_'):
            n_features = decision_tree.n_features_in_
        else:
            n_features = 100
        tree_feature_names = [f'F{i}' for i in range(n_features)]

    # 获取树的深度
    tree_depth = decision_tree.get_depth() if hasattr(decision_tree, 'get_depth') else 0

    # 根据深度调整图形大小
    if max_depth <= 4:
        figsize = (16, 12)
        fontsize = 10
    elif max_depth <= 6:
        figsize = (20, 15)
        fontsize = 9
    else:
        figsize = (25, 18)
        fontsize = 8

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    try:
        # 绘制决策树 - 最小化节点信息
        plot_tree(
            decision_tree,
            feature_names=tree_feature_names,
            class_names=['0', '1'],  # 使用0和1表示类别
            filled=True,
            rounded=True,
            fontsize=fontsize,
            ax=ax,
            max_depth=max_depth,
            impurity=False,  # 不显示基尼系数
            label='none',  # 不显示任何额外标签
            node_ids=False,  # 不显示节点ID
            proportion=False,  # 不显示样本比例
        )

        # 设置标题
        depth_info = f" (显示深度: {max_depth})" if max_depth < tree_depth else ""
        ax.set_title(f"决策树 - {project_name}{depth_info}",
                     fontsize=14, fontweight='bold', pad=20)

        # 自定义图例
        legend_elements = [
            mpatches.Patch(color='#FFE4B5', label='类别: 0 (无缺陷)'),
            mpatches.Patch(color='#87CEEB', label='类别: 1 (有缺陷)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # 添加说明文字
        info_text = f"树深度: {tree_depth}\n叶子节点数: {decision_tree.get_n_leaves()}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    except Exception as e:
        print(f"绘制决策树时出错: {e}")
        ax.text(0.5, 0.5, f'无法绘制决策树: {e}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("决策树 - 绘制失败", fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, f'decision_tree_minimal_depth{max_depth}.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"最小化决策树可视化已保存至: {output_path}")

    plt.close(fig)
    return fig


def visualize_decision_tree_binary(decision_tree, feature_names, project_name, max_depth=5):
    """
    二值化决策树可视化 - 节点只显示0或1

    Args:
        decision_tree: 决策树对象
        feature_names: 特征名称列表
        project_name: 项目名称
        max_depth: 最大显示深度
    """
    if decision_tree is None:
        print("警告：没有可用的决策树对象")
        return None

    # 创建输出目录
    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # 准备特征名称列表
    if feature_names is not None:
        tree_feature_names = feature_names
    else:
        if hasattr(decision_tree, 'n_features_in_'):
            n_features = decision_tree.n_features_in_
        else:
            n_features = 100
        tree_feature_names = [f'F{i}' for i in range(n_features)]

    # 创建图表
    fig, ax = plt.subplots(figsize=(18, 12))

    try:
        # 获取决策树结构信息
        n_nodes = decision_tree.tree_.node_count
        children_left = decision_tree.tree_.children_left
        children_right = decision_tree.tree_.children_right
        feature = decision_tree.tree_.feature
        threshold = decision_tree.tree_.threshold
        value = decision_tree.tree_.value

        # 计算每个节点的类别（0或1）
        node_class = []
        for i in range(n_nodes):
            if children_left[i] == children_right[i]:  # 叶子节点
                # 获取该节点的样本分布
                class_distribution = value[i][0]
                # 选择样本数最多的类别
                predicted_class = np.argmax(class_distribution)
                node_class.append(str(predicted_class))
            else:  # 内部节点
                # 内部节点不显示类别，显示特征条件
                feature_name = tree_feature_names[feature[i]]
                node_class.append(f"{feature_name}")

        # 绘制决策树
        plot_tree(
            decision_tree,
            feature_names=tree_feature_names,
            class_names=['0', '1'],
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax,
            max_depth=max_depth,
            impurity=False,
            label='none',
            node_ids=False,
            proportion=False,
        )

        # 修改节点文本：叶子节点显示0/1，内部节点显示特征名
        # 获取所有的文本对象
        text_objects = ax.texts

        # 我们需要解析每个文本对象的内容
        # 由于sklearn的plot_tree固定格式，我们可以用自定义的方式重新绘制

        # 清空当前图形
        ax.clear()

        # 使用更灵活的方式：导出为文本格式，然后解析
        tree_text = export_text(
            decision_tree,
            feature_names=tree_feature_names,
            max_depth=max_depth,
            decimals=1,
            show_weights=False
        )

        # 绘制简单的树形图
        ax.text(0.5, 0.5, "决策树结构（文本格式）\n\n" + tree_text,
                fontsize=9, family='monospace',
                ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ax.set_title(f"决策树（文本格式）- {project_name}",
                     fontsize=14, fontweight='bold', pad=20)

    except Exception as e:
        print(f"绘制二值化决策树时出错: {e}")
        # 回退到简单绘制
        try:
            plot_tree(
                decision_tree,
                feature_names=tree_feature_names,
                class_names=['0', '1'],
                filled=True,
                rounded=True,
                fontsize=10,
                ax=ax,
                max_depth=max_depth,
                impurity=False,
                label='none',
                node_ids=False,
                proportion=False,
            )
            ax.set_title(f"决策树 - {project_name}", fontsize=14, fontweight='bold')
        except Exception as e2:
            print(f"回退绘制也失败: {e2}")
            ax.text(0.5, 0.5, f'无法绘制决策树: {e2}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("决策树 - 绘制失败", fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, f'decision_tree_binary_depth{max_depth}.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f"二值化决策树可视化已保存至: {output_path}")

    plt.close(fig)
    return fig


def visualize_decision_tree_ultra_simple(decision_tree, feature_names, project_name):
    """
    超简化决策树可视化 - 只显示决策路径

    Args:
        decision_tree: 决策树对象
        feature_names: 特征名称列表
        project_name: 项目名称
    """
    if decision_tree is None:
        print("警告：没有可用的决策树对象")
        return None

    # 创建输出目录
    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # 获取决策规则
    from sklearn.tree import export_text

    if feature_names is not None:
        tree_feature_names = feature_names
    else:
        if hasattr(decision_tree, 'n_features_in_'):
            n_features = decision_tree.n_features_in_
        else:
            n_features = 100
        tree_feature_names = [f'F{i}' for i in range(n_features)]

    # 导出决策树文本
    tree_rules = export_text(
        decision_tree,
        feature_names=tree_feature_names,
        max_depth=10,
        decimals=1,
        show_weights=False
    )

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 显示决策规则
    ax.text(0.5, 0.5, f"决策树规则 - {project_name}\n\n{tree_rules}",
            fontsize=9, family='monospace',
            ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')

    # 设置标题
    tree_depth = decision_tree.get_depth() if hasattr(decision_tree, 'get_depth') else '未知'
    ax.set_title(f"决策树规则 (深度: {tree_depth})", fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'decision_tree_rules.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"决策树规则已保存至: {output_path}")

    # 同时保存为文本文件
    text_path = os.path.join(output_dir, 'decision_tree_rules.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f"决策树规则 - {project_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(tree_rules)
    print(f"决策树规则文本已保存至: {text_path}")

    plt.close(fig)
    return fig


def visualize_feature_importance_minimal(importances, feature_names, project_name):
    """
    最小化特征重要性可视化

    Args:
        importances: 特征重要性数组
        feature_names: 特征名称列表
        project_name: 项目名称
    """
    if importances is None:
        print("错误：没有可用的特征重要性数据")
        return None

    # 创建输出目录
    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # 只显示重要性大于0的特征
    nonzero_mask = importances > 0
    if np.sum(nonzero_mask) == 0:
        print("警告：没有找到重要的特征")
        return None

    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_importances = importances[nonzero_indices]

    # 按重要性排序
    sorted_idx = np.argsort(nonzero_importances)[::-1]

    # 限制显示的特征数量
    max_features = min(15, len(sorted_idx))
    top_indices = sorted_idx[:max_features]

    # 获取特征名称
    if feature_names is not None:
        feature_labels = [feature_names[i] for i in nonzero_indices[top_indices]]
    else:
        feature_labels = [f'F{i}' for i in nonzero_indices[top_indices]]

    top_importances = nonzero_importances[top_indices]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, max(6, max_features * 0.25)))

    # 创建水平条形图
    y_pos = np.arange(max_features)
    colors = ['#FF6B6B' if imp > 0.1 else '#4ECDC4' if imp > 0.05 else '#45B7D1' for imp in top_importances]

    bars = ax.barh(y_pos, top_importances, color=colors, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('重要性', fontsize=10)
    ax.set_title(f'特征重要性 - {project_name}',
                 fontsize=12, fontweight='bold', pad=15)

    # 添加数值标签
    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{importance:.3f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'feature_importance_minimal.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"特征重要性可视化已保存至: {output_path}")

    plt.close(fig)
    return fig


def create_decision_tree_summary(decision_tree, feature_names, importances, project_name):
    """
    创建决策树摘要

    Args:
        decision_tree: 决策树对象
        feature_names: 特征名称列表
        importances: 特征重要性数组
        project_name: 项目名称
    """
    if decision_tree is None:
        return

    # 创建输出目录
    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # 获取树的基本信息
    tree_depth = decision_tree.get_depth() if hasattr(decision_tree, 'get_depth') else '未知'
    n_leaves = decision_tree.get_n_leaves() if hasattr(decision_tree, 'get_n_leaves') else '未知'
    n_features = decision_tree.n_features_in_ if hasattr(decision_tree, 'n_features_in_') else '未知'

    # 获取最重要的特征
    top_features = []
    if importances is not None and feature_names is not None:
        # 获取前5个重要特征
        if len(importances) > 0:
            top_indices = np.argsort(importances)[-5:][::-1]
            top_features = [(feature_names[i] if i < len(feature_names) else f'F{i}',
                             importances[i]) for i in top_indices if importances[i] > 0]

    # 创建摘要文本
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"决策树摘要 - {project_name}")
    summary_lines.append("=" * 60)
    summary_lines.append(f"树深度: {tree_depth}")
    summary_lines.append(f"叶子节点数: {n_leaves}")
    summary_lines.append(f"特征数量: {n_features}")
    summary_lines.append("")

    if top_features:
        summary_lines.append("Top 5 重要特征:")
        summary_lines.append("-" * 40)
        for i, (feature, importance) in enumerate(top_features, 1):
            summary_lines.append(f"{i}. {feature}: {importance:.4f}")

    # 获取简单的决策规则
    try:
        from sklearn.tree import export_text
        if feature_names is not None:
            tree_feature_names = feature_names
        else:
            tree_feature_names = [f'F{i}' for i in range(n_features if isinstance(n_features, int) else 10)]

        tree_rules = export_text(
            decision_tree,
            feature_names=tree_feature_names,
            max_depth=3,
            decimals=1,
            show_weights=False
        )

        summary_lines.append("")
        summary_lines.append("主要决策规则 (前3层):")
        summary_lines.append("-" * 40)
        summary_lines.extend(tree_rules.split('\n')[:20])  # 只取前20行

    except Exception as e:
        summary_lines.append(f"无法获取决策规则: {e}")

    # 保存摘要
    summary_path = os.path.join(output_dir, 'decision_tree_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f"决策树摘要已保存至: {summary_path}")

    # 打印到控制台
    print('\n'.join(summary_lines[:20]))  # 只打印前20行


def process_project_minimal(project_name):
    """
    最小化版本项目处理函数

    Args:
        project_name: 项目名称
    """
    print(f"\n{'=' * 60}")
    print(f"处理项目: {project_name}")
    print(f"{'=' * 60}")

    try:
        # 1. 加载数据
        report_data = load_report_data(project_name)

        # 2. 提取特征信息
        decision_tree, feature_names, importances, importances_df = extract_feature_info(report_data)

        print(f"\n提取的信息:")
        print(f"- 决策树: {'可用' if decision_tree is not None else '不可用'}")
        if decision_tree is not None:
            if hasattr(decision_tree, 'get_depth'):
                print(f"  树深度: {decision_tree.get_depth()}")
            if hasattr(decision_tree, 'get_n_leaves'):
                print(f"  叶子节点数: {decision_tree.get_n_leaves()}")

        print(f"- 特征数量: {len(feature_names) if feature_names else '未知'}")
        print(f"- 特征重要性: {'可用' if importances is not None else '不可用'}")
        if importances is not None:
            print(f"  非零重要性特征数: {np.sum(importances > 0)}")

        # 3. 生成决策树可视化（最小化版本）
        if decision_tree is not None:
            # 生成多个深度的视图
            for depth in [3, 4, 5]:
                visualize_decision_tree_minimal(decision_tree, feature_names, project_name, max_depth=depth)

            # 生成超简化版本
            visualize_decision_tree_ultra_simple(decision_tree, feature_names, project_name)

            # 创建决策树摘要
            create_decision_tree_summary(decision_tree, feature_names, importances, project_name)

        # 4. 生成特征重要性可视化
        if importances is not None:
            visualize_feature_importance_minimal(importances, feature_names, project_name)

        # 5. 保存特征重要性数据
        if importances_df is not None:
            output_dir = f'./decision_tree/{project_name}/'
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, 'feature_importances.csv')
            importances_df.to_csv(csv_path, encoding='utf-8')
            print(f"特征重要性数据已保存至: {csv_path}")

        print(f"\n✓ 项目 '{project_name}' 处理完成")

    except FileNotFoundError as e:
        print(f"✗ 文件未找到: {e}")
    except Exception as e:
        print(f"✗ 处理项目 '{project_name}' 时发生错误: {e}")
        import traceback
        traceback.print_exc()

    print(f"{'=' * 60}\n")


def main():
    """
    主函数：处理所有项目
    """
    # 设置matplotlib样式
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 设置全局样式
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10

    print("开始生成最小化决策树可视化...")

    # 示例项目列表
    project_list = [
        'activemq-5.0.0',
        'wicket-1.5.3',
    ]

    # 自动发现项目
    auto_discover = True

    if auto_discover:
        report_dir = '../result_20251016/clustering/INTC_K-means/'
        if os.path.exists(report_dir):
            print(f"\n在目录中自动发现项目: {report_dir}")
            discovered_projects = []
            for file in os.listdir(report_dir):
                if file.endswith('.pkl'):
                    project_name = file.replace('.pkl', '')
                    if '_' in project_name:
                        base_name = project_name.split('_')[0]
                        if base_name not in discovered_projects:
                            discovered_projects.append(base_name)
                    else:
                        if project_name not in discovered_projects:
                            discovered_projects.append(project_name)

            if discovered_projects:
                project_list = discovered_projects
                print(f"发现 {len(project_list)} 个项目")
            else:
                print("未发现项目文件，使用示例项目列表")
        else:
            print(f"目录不存在: {report_dir}")
            print("使用示例项目列表")

    print(f"\n将要处理的项目: {project_list[:5]}...")  # 只显示前5个

    # 处理每个项目
    for project in project_list:
        process_project_minimal(project)

    print("\n" + "=" * 60)
    print("所有项目处理完成！")
    print("可视化文件保存在: ./decision_tree/ 目录下")
    print("=" * 60)


if __name__ == '__main__':
    main()