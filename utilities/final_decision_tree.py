import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pickle
import os
import glob
import numpy as np
import pandas as pd


# 绘图函数
def plot_decision_tree_sklearn(clf, feature_names, output_filename, figsize=(20, 10)):
    """可视化决策树"""
    fig, ax = plt.subplots(figsize=figsize)

    # 自定义类别名称：将0改为Non-defective，1改为Defective
    # 首先检查模型有哪些类别
    if hasattr(clf, 'classes_'):
        # 创建类别名称映射
        class_names = []
        for class_label in clf.classes_:
            if class_label == 0:
                class_names.append('Non-defective')
            elif class_label == 1:
                class_names.append('Defective')
            else:
                # 如果有更多类别，使用原始标签
                class_names.append(str(class_label))
    else:
        # 如果模型没有classes_属性，假设是二分类
        class_names = ['Non-defective', 'Defective']

    print(f"  类别名称映射: {dict(zip(range(len(class_names)), class_names))}")

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

    # 提取项目名和版本号
    if '-' in project_name:
        parts = project_name.split('-', 1)
        project = parts[0]
        version = parts[1] if len(parts) > 1 else ""
        title = f"{project} {version}"
    else:
        title = f"{project_name}"

    plt.title(title, fontsize=16, pad=20)

    # 添加子标题说明
    # plt.figtext(0.5, 0.01,
    #             f"Features: {len(feature_names)} | Classes: {', '.join(class_names)}",
    #             ha="center", fontsize=10, style='italic')

    plt.autoscale(enable=True, axis='both', tight=True)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 决策树已保存: {output_filename}")


def extract_real_feature_names(feature_importances, latent_dim):
    """从特征重要性中提取真实的特征名称"""
    if feature_importances is None:
        return None

    try:
        # 如果feature_importances是pandas Series或DataFrame
        if hasattr(feature_importances, 'index'):
            # 获取索引作为特征名称
            indices = feature_importances.index

            # 如果是多层索引（如(0,)），将其转换为字符串
            if len(indices) > 0:
                feature_names = []
                for idx in indices:
                    if isinstance(idx, tuple):
                        # 将元组转换为字符串，例如(0,) -> "feature_0"
                        if len(idx) == 1:
                            feature_names.append(f"feature_{idx[0]}")
                        else:
                            # 如果有多层，用下划线连接
                            feature_names.append('_'.join(str(i) for i in idx))
                    else:
                        feature_names.append(str(idx))

                # 确保特征名称数量与latent_dim匹配
                if len(feature_names) == latent_dim:
                    print(f"  从feature_importances中提取了 {len(feature_names)} 个特征名称")
                    print(f"  前5个特征名称: {feature_names[:5]}")
                    return feature_names
                else:
                    print(f"  警告: 提取的特征数量({len(feature_names)})与latent_dim({latent_dim})不匹配")

        # 如果feature_importances是numpy数组但没有索引信息
        elif isinstance(feature_importances, np.ndarray):
            print("  feature_importances是numpy数组，没有特征名称信息")

    except Exception as e:
        print(f"  提取特征名称时出错: {e}")

    return None


# 主程序
def main():
    print("=" * 60)
    print("批量可视化决策树工具")
    print("=" * 60)
    print("说明：类别标签映射 - 0: Non-defective, 1: Defective")
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
    output_dir = 'final_decision_tree'
    os.makedirs(output_dir, exist_ok=True)

    # 统计信息
    success_count = 0
    fail_count = 0

    # 遍历所有文件
    for i, file_path in enumerate(pkl_files):
        file_name = os.path.basename(file_path)
        print(f"\n{'=' * 50}")
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

            # 3. 获取模型信息
            if hasattr(tree_model, 'classes_'):
                print(f"  模型类别: {tree_model.classes_}")
                print(f"  类别数量: {len(tree_model.classes_)}")

            # 4. 获取特征维度
            latent_dim = data.get('latent_dim', 65)
            print(f"  特征维度: {latent_dim}")

            # 5. 从feature_importances中提取真实的特征名称
            feature_names = None
            if 'feature_importances' in data and data['feature_importances'] is not None:
                print("  尝试从feature_importances中提取真实的特征名称...")
                feature_names = extract_real_feature_names(data['feature_importances'], latent_dim)

            # 6. 如果无法提取真实特征名称，使用特征重要性索引或默认特征名
            if feature_names is None:
                print("  无法提取真实特征名称，使用特征重要性索引或默认特征名")

                # 检查特征重要性的类型
                if 'feature_importances' in data and data['feature_importances'] is not None:
                    importances = data['feature_importances']

                    # 如果是pandas Series/DataFrame，使用索引
                    if hasattr(importances, 'index'):
                        indices = importances.index
                        if len(indices) == latent_dim:
                            feature_names = [f"feature_{i}" for i in range(latent_dim)]
                        else:
                            # 创建基于特征重要性的特征名
                            feature_names = [f"feature_{i}" for i in range(latent_dim)]
                    else:
                        # 对于numpy数组，使用索引
                        feature_names = [f"feature_{i}" for i in range(latent_dim)]
                else:
                    # 如果没有特征重要性，使用默认特征名
                    feature_names = [f"feature_{i}" for i in range(latent_dim)]

            print(f"  使用的特征名称数量: {len(feature_names)}")
            print(f"  前5个特征名称: {feature_names[:5]}")

            # 7. 生成输出文件名
            base_name = os.path.splitext(file_name)[0]
            output_filename = os.path.join(output_dir, f"{base_name}_decision_tree.png")


            # 8. 可视化决策树
            plot_decision_tree_sklearn(tree_model, feature_names, output_filename)

            # 9. 打印额外信息
            try:
                if 'tree_max_depth' in data:
                    print(f"  决策树最大深度: {data['tree_max_depth']}")
                if 'tree_n_nodes' in data:
                    print(f"  决策树节点数: {data['tree_n_nodes']}")

                # 处理特征重要性
                if 'feature_importances' in data and data['feature_importances'] is not None:
                    importances = data['feature_importances']

                    # 转换特征重要性为numpy数组
                    if hasattr(importances, 'iloc'):  # pandas DataFrame/Series
                        importances_array = importances.values.flatten()
                    elif isinstance(importances, (list, np.ndarray)):
                        importances_array = np.array(importances).flatten()
                    else:
                        importances_array = None

                    if importances_array is not None and len(importances_array) > 0:
                        # 找到最重要的5个特征
                        top_n = min(5, len(importances_array))
                        top_indices = np.argsort(importances_array)[-top_n:][::-1]

                        print(f"  最重要的{top_n}个特征:")
                        for idx, importance in zip(top_indices, importances_array[top_indices]):
                            feature_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
                            print(f"    {feature_name}: {importance:.4f}")
            except Exception as info_error:
                print(f"  信息提取时遇到问题: {info_error}")

            success_count += 1

        except Exception as e:
            print(f"  ✗ 处理 {file_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
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
        for f in output_files[:10]:  # 只显示前10个
            file_size = os.path.getsize(f) / 1024  # KB
            print(f"  {os.path.basename(f)} ({file_size:.1f} KB)")

        if len(output_files) > 10:
            print(f"  ... 还有 {len(output_files) - 10} 个文件")

    # 生成汇总文件
    if success_count > 0:
        summary_file = os.path.join(output_dir, 'summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("决策树可视化汇总\n")
            f.write("=" * 50 + "\n")
            f.write(f"总文件数: {len(pkl_files)}\n")
            f.write(f"成功处理: {success_count}\n")
            f.write(f"失败处理: {fail_count}\n")
            f.write("类别标签映射: 0 -> Non-defective, 1 -> Defective\n")
            f.write("特征名称来源: 从pkl文件的feature_importances中提取\n")
            f.write("=" * 50 + "\n\n")

            # 重新读取每个文件，获取详细信息
            for file_path in pkl_files:
                file_name = os.path.basename(file_path)
                try:
                    with open(file_path, 'rb') as pf:
                        data = pickle.load(pf)

                    if isinstance(data, dict) and 'tree' in data:
                        tree_model = data['tree']
                        f.write(f"文件: {file_name}\n")

                        if hasattr(tree_model, 'classes_'):
                            f.write(f"  类别: {tree_model.classes_}\n")

                        if 'tree_max_depth' in data:
                            f.write(f"  最大深度: {data['tree_max_depth']}\n")
                        if 'tree_n_nodes' in data:
                            f.write(f"  节点数: {data['tree_n_nodes']}\n")

                        # 记录特征名称
                        latent_dim = data.get('latent_dim', 65)
                        f.write(f"  特征维度: {latent_dim}\n")

                        # 提取特征名称
                        feature_names = None
                        if 'feature_importances' in data and data['feature_importances'] is not None:
                            importances = data['feature_importances']
                            if hasattr(importances, 'index'):
                                indices = importances.index
                                if len(indices) > 0:
                                    f.write(f"  特征名称示例: {[str(idx) for idx in indices[:3]]}...\n")

                        f.write("\n")
                except:
                    f.write(f"文件: {file_name} - 读取失败\n\n")

        print(f"\n汇总信息已保存到: {summary_file}")


# 运行主程序
if __name__ == "__main__":
    main()