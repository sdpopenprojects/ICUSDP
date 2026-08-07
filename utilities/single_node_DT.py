import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pickle
import os
import glob
import numpy as np


# 绘图函数
def plot_decision_tree_sklearn(clf, feature_names, output_filename, figsize=(20, 10)):
    """可视化决策树"""
    fig, ax = plt.subplots(figsize=figsize)

    # 获取类别名称
    if hasattr(clf, 'classes_'):
        class_names = [str(i) for i in clf.classes_]
    else:
        class_names = None

    # 绘制决策树
    tree_plot = plot_tree(clf,
                          feature_names=feature_names,
                          class_names=class_names,
                          label='all',
                          filled=True,
                          impurity=False,
                          node_ids=False,
                          proportion=False,
                          rounded=True,
                          fontsize=8,
                          ax=ax)

    # 简化标签（移除gini等）
    for text in tree_plot:
        content = text.get_text()
        strs = content.split('\n')
        if len(strs) == 3:
            text.set_text(strs[-1])  # 只保留类别
        else:
            text.set_text(strs[0])  # 只保留节点条件

    # 设置标题
    base_name = os.path.splitext(os.path.basename(output_filename))[0]
    project_name = base_name.split('_decision_tree')[0]
    plt.title(f"Decision Tree - {project_name}", fontsize=16)

    plt.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 决策树已保存: {output_filename}")


# 主程序
def main():
    print("=" * 60)
    print("批量可视化决策树工具")
    print("=" * 60)

    # 设置路径
    base_path = '../result_20251016/clustering/INTC_K-means/'

    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"错误：路径不存在 - {base_path}")
        print("当前工作目录:", os.getcwd())
        return

    # 获取所有pkl文件
    pkl_files = sorted(glob.glob(os.path.join(base_path, '*.pkl')))

    if not pkl_files:
        print(f"错误：在指定路径下未找到pkl文件")
        return

    print(f"找到 {len(pkl_files)} 个pkl文件")

    # 创建输出目录
    output_dir = 'decision_trees_visualized'
    os.makedirs(output_dir, exist_ok=True)

    # 统计信息
    success_count = 0
    fail_count = 0

    # 遍历所有文件
    for i, file_path in enumerate(pkl_files):
        file_name = os.path.basename(file_path)
        print(f"\n{'=' * 40}")
        print(f"处理文件 {i + 1}/{len(pkl_files)}: {file_name}")

        try:
            # 1. 加载数据
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            print(f"  数据类型: {type(data)}")

            if not isinstance(data, dict):
                print(f"  ✗ 数据不是字典格式，跳过")
                fail_count += 1
                continue

            # 2. 提取决策树
            if 'tree' not in data:
                print(f"  ✗ 未找到决策树，跳过")
                fail_count += 1
                continue

            tree_model = data['tree']
            print(f"  ✓ 找到已训练的决策树模型")

            # 3. 获取特征维度并创建特征名
            latent_dim = data.get('latent_dim', 65)  # 默认65，从输出可见都是65
            feature_names = [f'latent_feature_{j}' for j in range(latent_dim)]
            print(f"  特征维度: {latent_dim}")

            # 4. 生成输出文件名
            base_name = os.path.splitext(file_name)[0]
            output_filename = os.path.join(output_dir, f"{base_name}_decision_tree.png")

            # 5. 可视化决策树
            plot_decision_tree_sklearn(tree_model, feature_names, output_filename)

            # 6. 打印额外信息（但不让错误中断程序）
            try:
                if 'tree_max_depth' in data:
                    print(f"  决策树最大深度: {data['tree_max_depth']}")
                if 'tree_n_nodes' in data:
                    print(f"  决策树节点数: {data['tree_n_nodes']}")

                # 处理特征重要性（更安全的方式）
                if 'feature_importances' in data and data['feature_importances'] is not None:
                    importances = data['feature_importances']

                    # 检查importances的类型
                    if hasattr(importances, 'iloc'):  # 如果是pandas DataFrame/Series
                        # 转换为numpy数组
                        importances_array = importances.values.flatten()
                        print(f"  特征重要性类型: pandas, 形状: {importances.shape}")
                    elif isinstance(importances, (list, np.ndarray)):
                        importances_array = np.array(importances).flatten()
                        print(f"  特征重要性类型: {type(importances).__name__}, 长度: {len(importances_array)}")
                    else:
                        print(f"  特征重要性类型: {type(importances)}")
                        importances_array = None

                    if importances_array is not None and len(importances_array) > 0:
                        # 找到最重要的3个特征
                        non_zero_idx = np.where(importances_array > 0)[0]
                        if len(non_zero_idx) > 0:
                            top_n = min(3, len(non_zero_idx))
                            # 按重要性排序
                            sorted_idx = np.argsort(importances_array[non_zero_idx])[-top_n:][::-1]
                            top_features = non_zero_idx[sorted_idx]
                            top_importances = importances_array[non_zero_idx][sorted_idx]

                            print(f"  最重要的{top_n}个特征: {top_features}")
                            print(f"  重要性值: {top_importances}")
                        else:
                            print(f"  所有特征重要性都为0")
            except Exception as info_error:
                print(f"  信息提取时遇到小问题: {info_error}")

            success_count += 1

        except Exception as e:
            print(f"  ✗ 处理 {file_name} 时出错: {str(e)}")
            fail_count += 1

    # 汇总报告
    print(f"\n{'=' * 60}")
    print("处理完成!")
    print(f"{'=' * 60}")
    print(f"总文件数: {len(pkl_files)}")
    print(f"成功处理: {success_count}")
    print(f"失败处理: {fail_count}")
    print(f"输出目录: {output_dir}")

    # 列出生成的文件
    if success_count > 0:
        print("\n生成的决策树文件:")
        output_files = sorted(glob.glob(os.path.join(output_dir, '*.png')))
        for f in output_files:
            file_size = os.path.getsize(f) / 1024  # KB
            print(f"  {os.path.basename(f)} ({file_size:.1f} KB)")


# 运行主程序
if __name__ == "__main__":
    main()