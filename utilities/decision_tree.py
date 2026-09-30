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
                raise FileNotFoundError(f": {report_file}")

    with open(report_file, 'rb') as f:
        report_data = pickle.load(f)

    print(f" '{project_name}' ")
    return report_data


def extract_feature_info(report_data):

    decision_tree = None
    if 'tree' in report_data:
        decision_tree = report_data['tree']
        print("")

    importances_df = None
    if 'feature_importances' in report_data and isinstance(report_data['feature_importances'], pd.DataFrame):
        importances_df = report_data['feature_importances']
        print("DataFrame")

    feature_names = None
    importances = None

    if importances_df is not None:
        feature_names = importances_df.index.tolist()
        if len(importances_df.columns) == 1:
            importances = importances_df.iloc[:, 0].values
        else:
            importances = importances_df.iloc[:, 0].values
            print(f" {len(importances_df.columns)} ")

    if importances is None and decision_tree is not None:
        if hasattr(decision_tree, 'feature_importances_'):
            importances = decision_tree.feature_importances_
            print("")

    if feature_names is None and importances is not None:
        n_features = len(importances)
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        print(f" {n_features} ")

    return decision_tree, feature_names, importances, importances_df


def visualize_decision_tree_minimal(decision_tree, feature_names, project_name, max_depth=5):

    if decision_tree is None:
        return None


    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)


    if feature_names is not None:
        tree_feature_names = feature_names
    else:
        if hasattr(decision_tree, 'n_features_in_'):
            n_features = decision_tree.n_features_in_
        else:
            n_features = 100
        tree_feature_names = [f'F{i}' for i in range(n_features)]


    tree_depth = decision_tree.get_depth() if hasattr(decision_tree, 'get_depth') else 0


    if max_depth <= 4:
        figsize = (16, 12)
        fontsize = 10
    elif max_depth <= 6:
        figsize = (20, 15)
        fontsize = 9
    else:
        figsize = (25, 18)
        fontsize = 8


    fig, ax = plt.subplots(figsize=figsize)

    try:
        plot_tree(
            decision_tree,
            feature_names=tree_feature_names,
            class_names=['0', '1'],
            filled=True,
            rounded=True,
            fontsize=fontsize,
            ax=ax,
            max_depth=max_depth,
            impurity=False,
            label='none',
            node_ids=False,
            proportion=False,
        )


        depth_info = f" (: {max_depth})" if max_depth < tree_depth else ""
        ax.set_title(f" - {project_name}{depth_info}",
                     fontsize=14, fontweight='bold', pad=20)


        legend_elements = [
            mpatches.Patch(color='#FFE4B5', label=': 0 '),
            mpatches.Patch(color='#87CEEB', label=': 1 '),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        info_text = f": {tree_depth}\n node: {decision_tree.get_n_leaves()}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    except Exception as e:
        print(f": {e}")
        ax.text(0.5, 0.5, f': {e}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("fail", fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'decision_tree_minimal_depth{max_depth}.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f": {output_path}")

    plt.close(fig)
    return fig


def visualize_decision_tree_binary(decision_tree, feature_names, project_name, max_depth=5):

    if decision_tree is None:

        return None


    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)


    if feature_names is not None:
        tree_feature_names = feature_names
    else:
        if hasattr(decision_tree, 'n_features_in_'):
            n_features = decision_tree.n_features_in_
        else:
            n_features = 100
        tree_feature_names = [f'F{i}' for i in range(n_features)]


    fig, ax = plt.subplots(figsize=(18, 12))

    try:

        n_nodes = decision_tree.tree_.node_count
        children_left = decision_tree.tree_.children_left
        children_right = decision_tree.tree_.children_right
        feature = decision_tree.tree_.feature
        threshold = decision_tree.tree_.threshold
        value = decision_tree.tree_.value


        node_class = []
        for i in range(n_nodes):
            if children_left[i] == children_right[i]:

                class_distribution = value[i][0]

                predicted_class = np.argmax(class_distribution)
                node_class.append(str(predicted_class))
            else:

                feature_name = tree_feature_names[feature[i]]
                node_class.append(f"{feature_name}")


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


        text_objects = ax.texts


        ax.clear()


        tree_text = export_text(
            decision_tree,
            feature_names=tree_feature_names,
            max_depth=max_depth,
            decimals=1,
            show_weights=False
        )


        ax.text(0.5, 0.5, "）\n\n" + tree_text,
                fontsize=9, family='monospace',
                ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        ax.set_title(f"- {project_name}",
                     fontsize=14, fontweight='bold', pad=20)

    except Exception as e:
        print(f": {e}")

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
            ax.set_title(f" - {project_name}", fontsize=14, fontweight='bold')
        except Exception as e2:
            print(f": {e2}")
            ax.text(0.5, 0.5, f': {e2}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("fail", fontsize=14, fontweight='bold')

    plt.tight_layout()


    output_path = os.path.join(output_dir, f'decision_tree_binary_depth{max_depth}.png')
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    print(f": {output_path}")

    plt.close(fig)
    return fig


def visualize_decision_tree_ultra_simple(decision_tree, feature_names, project_name):

    if decision_tree is None:
        return None


    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)


    from sklearn.tree import export_text

    if feature_names is not None:
        tree_feature_names = feature_names
    else:
        if hasattr(decision_tree, 'n_features_in_'):
            n_features = decision_tree.n_features_in_
        else:
            n_features = 100
        tree_feature_names = [f'F{i}' for i in range(n_features)]


    tree_rules = export_text(
        decision_tree,
        feature_names=tree_feature_names,
        max_depth=10,
        decimals=1,
        show_weights=False
    )


    fig, ax = plt.subplots(figsize=(12, 8))


    ax.text(0.5, 0.5, f" - {project_name}\n\n{tree_rules}",
            fontsize=9, family='monospace',
            ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')


    tree_depth = decision_tree.get_depth() if hasattr(decision_tree, 'get_depth') else 'no know'
    ax.set_title(f" {tree_depth}", fontsize=14, fontweight='bold')

    plt.tight_layout()


    output_path = os.path.join(output_dir, 'decision_tree_rules.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"{output_path}")


    text_path = os.path.join(output_dir, 'decision_tree_rules.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(f" - {project_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(tree_rules)
    print(f": {text_path}")

    plt.close(fig)
    return fig


def visualize_feature_importance_minimal(importances, feature_names, project_name):

    if importances is None:
        return None


    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)


    nonzero_mask = importances > 0
    if np.sum(nonzero_mask) == 0:
        return None

    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_importances = importances[nonzero_indices]


    sorted_idx = np.argsort(nonzero_importances)[::-1]


    max_features = min(15, len(sorted_idx))
    top_indices = sorted_idx[:max_features]


    if feature_names is not None:
        feature_labels = [feature_names[i] for i in nonzero_indices[top_indices]]
    else:
        feature_labels = [f'F{i}' for i in nonzero_indices[top_indices]]

    top_importances = nonzero_importances[top_indices]


    fig, ax = plt.subplots(figsize=(10, max(6, max_features * 0.25)))


    y_pos = np.arange(max_features)
    colors = ['#FF6B6B' if imp > 0.1 else '#4ECDC4' if imp > 0.05 else '#45B7D1' for imp in top_importances]

    bars = ax.barh(y_pos, top_importances, color=colors, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel( fontsize=10)
    ax.set_title(f' - {project_name}',
                 fontsize=12, fontweight='bold', pad=15)


    for i, (bar, importance) in enumerate(zip(bars, top_importances)):
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{importance:.3f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(output_dir, 'feature_importance_minimal.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f": {output_path}")

    plt.close(fig)
    return fig


def create_decision_tree_summary(decision_tree, feature_names, importances, project_name):

    if decision_tree is None:
        return


    output_dir = f'./decision_tree/{project_name}/'
    os.makedirs(output_dir, exist_ok=True)


    tree_depth = decision_tree.get_depth() if hasattr(decision_tree, 'get_depth') else '未知'
    n_leaves = decision_tree.get_n_leaves() if hasattr(decision_tree, 'get_n_leaves') else '未知'
    n_features = decision_tree.n_features_in_ if hasattr(decision_tree, 'n_features_in_') else '未知'


    top_features = []
    if importances is not None and feature_names is not None:

        if len(importances) > 0:
            top_indices = np.argsort(importances)[-5:][::-1]
            top_features = [(feature_names[i] if i < len(feature_names) else f'F{i}',
                             importances[i]) for i in top_indices if importances[i] > 0]

    # 创建摘要文本
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"summary - {project_name}")
    summary_lines.append("=" * 60)
    summary_lines.append(f"tree depth: {tree_depth}")
    summary_lines.append(f"node: {n_leaves}")
    summary_lines.append(f": {n_features}")
    summary_lines.append("")

    if top_features:
        summary_lines.append("Top 5 :")
        summary_lines.append("-" * 40)
        for i, (feature, importance) in enumerate(top_features, 1):
            summary_lines.append(f"{i}. {feature}: {importance:.4f}")


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
        summary_lines.append(":")
        summary_lines.append("-" * 40)
        summary_lines.extend(tree_rules.split('\n')[:20])

    except Exception as e:
        summary_lines.append(f": {e}")


    summary_path = os.path.join(output_dir, 'decision_tree_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f"save: {summary_path}")


    print('\n'.join(summary_lines[:20]))


def process_project_minimal(project_name):

    print(f"\n{'=' * 60}")
    print(f"处理项目: {project_name}")
    print(f"{'=' * 60}")

    try:

        report_data = load_report_data(project_name)


        decision_tree, feature_names, importances, importances_df = extract_feature_info(report_data)


        print(f"- : {'' if decision_tree is not None else ''}")
        if decision_tree is not None:
            if hasattr(decision_tree, 'get_depth'):
                print(f"  depth: {decision_tree.get_depth()}")
            if hasattr(decision_tree, 'get_n_leaves'):
                print(f"  node: {decision_tree.get_n_leaves()}")

        print(f"- : {len(feature_names) if feature_names else ''}")
        print(f"- : {'' if importances is not None else ''}")
        if importances is not None:
            print(f": {np.sum(importances > 0)}")


        if decision_tree is not None:

            for depth in [3, 4, 5]:
                visualize_decision_tree_minimal(decision_tree, feature_names, project_name, max_depth=depth)


            visualize_decision_tree_ultra_simple(decision_tree, feature_names, project_name)


            create_decision_tree_summary(decision_tree, feature_names, importances, project_name)


        if importances is not None:
            visualize_feature_importance_minimal(importances, feature_names, project_name)


        if importances_df is not None:
            output_dir = f'./decision_tree/{project_name}/'
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, 'feature_importances.csv')
            importances_df.to_csv(csv_path, encoding='utf-8')
            print(f": {csv_path}")

        print(f"\n✓  '{project_name}' ")

    except FileNotFoundError as e:
        print(f"✗ : {e}")
    except Exception as e:
        print(f"✗  '{project_name}' : {e}")
        import traceback
        traceback.print_exc()

    print(f"{'=' * 60}\n")


def main():


    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10




    project_list = [
        'activemq-5.0.0',
        'wicket-1.5.3',
    ]


    auto_discover = True

    if auto_discover:
        report_dir = '../result_20251016/clustering/INTC_K-means/'
        if os.path.exists(report_dir):
            print(f"\n: {report_dir}")
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
                print(f" {len(project_list)} ")
            else:
                print("no")
        else:
            print(f": {report_dir}")
            print("")

    print(f"\n: {project_list[:5]}...")  # 只显示前5个


    for project in project_list:
        process_project_minimal(project)

    print("\n" + "=" * 60)
    print("=" * 60)


if __name__ == '__main__':
    main()