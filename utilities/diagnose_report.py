# diagnose_report.py - 修复版
import os
import numpy as np
from utilities.File import load_results_pickle_v2

# 修改路径，根据您之前的文件夹结构
path = '../result_20251016/clustering/'
model_name = 'INTC_K-means'

full_path = os.path.join(path, model_name)

if os.path.exists(full_path):
    files = [f for f in sorted(os.listdir(full_path)) if f.endswith('.pkl')]

    if files:
        # 只分析第一个文件
        file_path = os.path.join(full_path, files[0])
        print(f"分析文件: {files[0]}")

        reports = load_results_pickle_v2(file_path)

        print(f"报告数量: {len(reports)}")

        if reports:
            first_report = reports[0]
            print(f"\n第一个报告的类型: {type(first_report)}")

            if isinstance(first_report, dict):
                print("报告中的键:")
                for key in first_report.keys():
                    value = first_report[key]
                    print(f"  '{key}': {type(value)}", end="")

                    # 显示一些基本信息
                    if isinstance(value, (int, float, str, bool)):
                        print(f" = {value}")
                    elif isinstance(value, (list, np.ndarray)):
                        print(f" (长度: {len(value)})")
                    elif hasattr(value, '__len__'):
                        print(f" (长度: {len(value)})")
                    else:
                        print()

                print("\n" + "=" * 60)
                print("深入分析 'tree' 键:")
                print("=" * 60)

                if 'tree' in first_report:
                    tree = first_report['tree']
                    print(f"树对象类型: {type(tree)}")

                    # 如果是sklearn的决策树，可以获取一些属性
                    if hasattr(tree, 'max_depth'):
                        print(f"树最大深度: {tree.max_depth}")
                    if hasattr(tree, 'n_features_in_'):
                        print(f"输入特征数: {tree.n_features_in_}")
                    if hasattr(tree, 'feature_importances_'):
                        print(f"特征重要性形状: {tree.feature_importances_.shape}")
                    if hasattr(tree, 'tree_'):
                        print(f"树节点数: {tree.tree_.node_count}")

                # 检查是否有其他可能的键
                expected_keys = ['feature_importances', 'pseudo_labels', 'latent_dim',
                                 'n_vaes', 'view_importances', 'cluster_labels',
                                 'performance', 'metrics', 'measure']

                print(f"\n" + "=" * 60)
                print("检查其他预期的键:")
                print("=" * 60)

                for key in expected_keys:
                    if key in first_report:
                        value = first_report[key]
                        print(f"  ✓ '{key}': {type(value)}", end="")
                        if isinstance(value, (list, np.ndarray)) and hasattr(value, '__len__'):
                            print(f" (长度: {len(value)})")
                        elif isinstance(value, dict):
                            print(f" (字典键: {list(value.keys())[:5]}...)")
                        else:
                            print()
                    else:
                        print(f"  ✗ '{key}': 不存在")

            else:
                print(f"报告内容: {first_report}")
    else:
        print("没有找到PKL文件")
else:
    print(f"路径不存在: {full_path}")
    print(f"当前工作目录: {os.getcwd()}")