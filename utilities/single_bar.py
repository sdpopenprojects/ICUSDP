# merge_best_analysis_single_bar.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def setup_chinese_font():
    """配置中文字体"""
    try:
        # 尝试使用系统字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    import matplotlib.font_manager as fm
                    fm.fontManager.addfont(font_path)
                    font_name = fm.FontProperties(fname=font_path).get_name()

                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    return True, font_name
                except:
                    continue

        # 回退到英文
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        return False, "Arial"

    except Exception as e:
        print(f"字体设置异常: {e}")
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        return False, "Arial"


def load_all_pkl_files(base_dir='../result_20251016/clustering/INTC_K-means/'):
    """
    加载所有pkl文件数据

    Returns:
        all_data_list: 包含所有文件数据的列表
        common_features: 所有文件共有的特征集合
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"目录不存在: {base_dir}")

    # 获取所有pkl文件
    pkl_files = [f for f in os.listdir(base_dir) if f.endswith('.pkl')]
    pkl_files.sort()

    all_data_list = []
    all_feature_sets = []

    print(f"找到 {len(pkl_files)} 个pkl文件")
    print("-" * 70)

    for i, filename in enumerate(pkl_files, 1):
        try:
            file_path = os.path.join(base_dir, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'feature_importances' in data:
                importances_df = data['feature_importances']

                if isinstance(importances_df, pd.DataFrame) and len(importances_df.columns) >= 1:
                    # 获取特征名称和重要性值
                    feature_names = importances_df.index.tolist()
                    importances = importances_df.iloc[:, 0].values

                    # 创建数据记录
                    data_record = {
                        'filename': filename,
                        'project_name': filename.replace('.pkl', ''),
                        'importances': importances,
                        'feature_names': feature_names
                    }

                    all_data_list.append(data_record)
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
        common_features = set.intersection(*all_feature_sets)
        print(f"\n成功加载 {len(all_data_list)} 个文件")
        print(f"共有特征数量: {len(common_features)}")
        return all_data_list, common_features
    else:
        print("没有成功加载任何数据")
        return None, None


def calculate_feature_statistics(all_data_list, common_features):
    """
    计算每个特征的统计量（基于所有非零重要性值）

    Args:
        all_data_list: 所有文件数据列表
        common_features: 共有特征集合

    Returns:
        df_stats: 包含特征统计量的DataFrame
    """
    # 创建特征到所有重要性值的映射
    feature_to_importances = defaultdict(list)

    for record in all_data_list:
        importances = record['importances']
        feature_names = record['feature_names']

        # 创建特征名到重要性的映射
        feature_to_importance = dict(zip(feature_names, importances))

        # 只收集非零重要性值
        for feature in common_features:
            if feature in feature_to_importance:
                importance = feature_to_importance[feature]
                if importance > 0:  # 只收集非零值
                    feature_to_importances[feature].append(importance)

    # 计算每个特征的统计量
    stats_data = []
    for feature in sorted(common_features):
        if feature in feature_to_importances:
            importances = feature_to_importances[feature]

            if importances:  # 确保有数据
                stats_data.append({
                    '特征': feature,
                    '平均重要性': np.mean(importances),
                    '中位数': np.median(importances),
                    '标准差': np.std(importances),
                    '出现次数': len(importances),
                    '最大重要性': np.max(importances),
                    '最小重要性': np.min(importances),
                    '变异系数': np.std(importances) / np.mean(importances) if np.mean(importances) > 0 else 0
                })

    # 创建DataFrame并排序
    df_stats = pd.DataFrame(stats_data)
    df_stats = df_stats.sort_values('平均重要性', ascending=False)

    return df_stats


def create_best_analysis_bar_chart(df_stats, n_files, save_dir='./best_analysis/'):
    """
    创建最佳分析方法的柱状图

    Args:
        df_stats: 特征统计DataFrame
        n_files: 文件总数
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 设置参数
    top_n = 20  # 显示前20个特征
    use_chinese, font_name = setup_chinese_font()

    # 准备数据
    top_features = df_stats.head(top_n)['特征'].tolist()
    top_avg_importances = df_stats.head(top_n)['平均重要性'].tolist()
    top_counts = df_stats.head(top_n)['出现次数'].tolist()
    top_stds = df_stats.head(top_n)['标准差'].tolist()

    # 创建图表
    plt.figure(figsize=(14, 10))

    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))

    # 创建水平条形图
    y_pos = np.arange(top_n)
    bars = plt.barh(y_pos, top_avg_importances, color=colors, height=0.7)

    # 添加误差条（标准差）
    plt.errorbar(top_avg_importances, y_pos,
                 xerr=top_stds,
                 fmt='none', ecolor='darkgray', alpha=0.7, capsize=3, linewidth=1.5)

    # 设置y轴
    plt.yticks(y_pos, top_features, fontsize=11)
    plt.gca().invert_yaxis()  # 最重要的特征在顶部

    # 设置x轴
    max_importance = max(top_avg_importances)
    plt.xlim(0, max_importance * 1.25)
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')

    # 添加数值标签
    for i, (bar, imp, count, std) in enumerate(zip(bars, top_avg_importances, top_counts, top_stds)):
        width = bar.get_width()

        # 在条形右侧添加标签
        label_text = f'{imp:.4f} ({count}/{n_files})'
        plt.text(width + 0.0005, bar.get_y() + bar.get_height() / 2,
                 label_text, ha='left', va='center', fontsize=10, fontweight='bold')

        # 在条形内部添加标准差信息（浅色）
        std_text = f'±{std:.4f}'
        if width > 0.02:  # 只在足够宽的条形内添加
            plt.text(width * 0.1, bar.get_y() + bar.get_height() / 2,
                     std_text, ha='left', va='center', fontsize=8,
                     color='white', fontweight='bold')

    # 设置标题和标签
    if use_chinese:
        plt.title(f'特征重要性Top {top_n} ',
                  fontsize=16, fontweight='bold', pad=20)
        # plt.xlabel('平均重要性分数（误差条表示标准差）', fontsize=13, fontweight='bold')
    else:
        plt.title(f'Feature Importance Top {top_n} (Based on {len(df_stats)} features, {n_files} files)',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Average Importance Score (error bars show standard deviation)',
                   fontsize=13, fontweight='bold')

    # 添加图例
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='darkgray', lw=2, label='标准差'),
        mpatches.Patch(color='gray', alpha=0.3, label=f'基于非零值计算 (n={n_files}个文件)')
    ]

    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()

    # 保存图表
    if use_chinese:
        chart_name = '特征重要性Top20.png'
    else:
        chart_name = 'feature_importance_top20.png'

    chart_path = os.path.join(save_dir, chart_name)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 柱状图已保存: {chart_path}")

    return chart_path


def save_statistics_and_report(df_stats, n_files, save_dir='./best_analysis/'):
    """
    保存统计数据和报告

    Args:
        df_stats: 特征统计DataFrame
        n_files: 文件总数
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存完整统计数据
    csv_path = os.path.join(save_dir, 'feature_statistics_complete.csv')
    df_stats.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ 完整统计数据已保存: {csv_path}")

    # 保存Top 20数据
    top_20_path = os.path.join(save_dir, 'top_20_features.csv')
    df_stats.head(20).to_csv(top_20_path, index=False, encoding='utf-8')
    print(f"✓ Top 20数据已保存: {top_20_path}")

    # 生成分析报告
    report_path = os.path.join(save_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("特征重要性分析报告（最佳分析方法）\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n")
        f.write(f"分析方法: 基于非零重要性值的统计\n")
        f.write(f"总文件数: {n_files}\n")
        f.write(f"分析特征数: {len(df_stats)}\n")
        f.write(f"零值处理: 排除零值，只计算特征实际出现时的重要性\n\n")

        f.write("Top 20 最重要的特征:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'排名':<4} {'特征':<30} {'平均重要性':<12} {'出现次数':<12} {'标准差':<12} {'变异系数':<12}\n")
        f.write("-" * 90 + "\n")

        for i, row in df_stats.head(20).iterrows():
            f.write(f"{i + 1:<4} {row['特征']:<30} {row['平均重要性']:<12.4f} "
                    f"{row['出现次数']:<12} {row['标准差']:<12.4f} {row['变异系数']:<12.4f}\n")

        f.write("\n分析说明:\n")
        f.write("1. 平均重要性: 基于特征在所有文件中实际出现时的非零重要性值计算\n")
        f.write("2. 出现次数: 特征在多少个文件中具有非零重要性\n")
        f.write("3. 标准差: 特征重要性值的离散程度，越小表示越稳定\n")
        f.write("4. 变异系数: 标准差/平均值，标准化后的离散程度\n")

    print(f"✓ 分析报告已保存: {report_path}")

    # 打印关键统计摘要
    print("\n" + "=" * 90)
    print("特征重要性分析摘要")
    print("=" * 90)
    print(f"总文件数: {n_files}")
    print(f"分析特征数: {len(df_stats)}")
    print(f"零值处理: 排除零值，只计算特征实际出现时的重要性\n")

    print("Top 10 最重要的特征:")
    print("-" * 90)
    print(f"{'排名':<4} {'特征':<30} {'平均重要性':<12} {'出现次数':<12} {'标准差':<12}")
    print("-" * 90)

    for i, row in df_stats.head(10).iterrows():
        print(f"{i + 1:<4} {row['特征']:<30} {row['平均重要性']:<12.4f} "
              f"{row['出现次数']:<12} {row['标准差']:<12.4f}")

    print("=" * 90)


def main():
    """
    主函数：执行最佳分析方法并生成单一柱状图
    """
    print("=" * 90)
    print("软件缺陷预测特征重要性分析 - 最佳方法")
    print("=" * 90)
    print("分析方法: 基于非零重要性值的统计")
    print("图表输出: 单一柱状图展示Top 20特征")
    print("=" * 90)

    try:
        # 1. 加载数据
        print("\n1. 加载数据...")
        base_dir = '../result_20251016/clustering/INTC_K-means/'
        all_data_list, common_features = load_all_pkl_files(base_dir)

        if not all_data_list:
            print("数据加载失败，程序退出")
            return

        n_files = len(all_data_list)

        # 2. 计算特征统计量（基于非零值）
        print("\n2. 计算特征统计量...")
        print("   方法: 只考虑特征实际出现时的非零重要性值")
        print("   优势: 避免零值对平均重要性的稀释效应")

        df_stats = calculate_feature_statistics(all_data_list, common_features)

        if df_stats.empty:
            print("统计计算失败，程序退出")
            return

        # 3. 创建最佳分析柱状图
        print("\n3. 创建柱状图...")
        save_dir = './best_analysis_single_bar/'
        chart_path = create_best_analysis_bar_chart(df_stats, n_files, save_dir)

        # 4. 保存统计数据和报告
        print("\n4. 保存统计数据和报告...")
        save_statistics_and_report(df_stats, n_files, save_dir)

        print(f"\n✓ 分析完成!")
        print(f"✓ 柱状图: {chart_path}")
        print(f"✓ 输出目录: {save_dir}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 90)
    print("程序执行完成!")
    print("=" * 90)


if __name__ == '__main__':
    main()