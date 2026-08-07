# final_chinese_chart_fixed.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def setup_chinese_font():
    """
    配置中文字体 - 修复版本
    """
    try:
        # 方法1：直接指定Windows中文字体路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/simkai.ttf',  # 楷体
            'C:/Windows/Fonts/simfang.ttf',  # 仿宋
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    # 将字体添加到matplotlib
                    matplotlib.font_manager.fontManager.addfont(font_path)
                    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()

                    # 设置matplotlib使用这个字体
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False

                    print(f"✓ 使用中文字体: {font_name} ({os.path.basename(font_path)})")
                    return True, font_name

                except Exception as e:
                    print(f"✗ 加载字体 {font_path} 失败: {e}")

        # 方法2：尝试使用已知的字体名称
        print("\n尝试使用已知字体名称...")
        chinese_font_names = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',  # 黑体
            'SimSun',  # 宋体
            'NSimSun',  # 新宋体
            'KaiTi',  # 楷体
            'FangSong',  # 仿宋
            'YouYuan',  # 幼圆
            'LiSu',  # 隶书
            'STXihei',  # 华文细黑
            'STHeiti',  # 华文黑体
            'STSong',  # 华文宋体
            'STKaiti',  # 华文楷体
            'STFangsong',  # 华文仿宋
        ]

        import matplotlib.font_manager as fm

        for font_name in chinese_font_names:
            try:
                # 尝试查找字体
                fm.findfont(font_name, fallback_to_default=False)
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False

                print(f"✓ 找到系统字体: {font_name}")
                return True, font_name
            except:
                continue

        print("⚠ 未找到中文字体，将使用英文")
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        return False, "Arial"

    except Exception as e:
        print(f"字体设置异常: {e}")
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        return False, "Arial"


def load_all_pkl_data():
    """加载所有pkl文件数据"""
    base_dir = '../result_20251016/clustering/INTC_K-means/'

    if not os.path.exists(base_dir):
        print(f"目录不存在: {base_dir}")
        return None, None, None

    pkl_files = [f for f in os.listdir(base_dir) if f.endswith('.pkl')]
    pkl_files.sort()

    print(f"找到 {len(pkl_files)} 个pkl文件")
    print("-" * 70)

    all_importances = []
    all_feature_names = []
    file_info = []

    for i, filename in enumerate(pkl_files, 1):
        try:
            file_path = os.path.join(base_dir, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'feature_importances' in data:
                importances_df = data['feature_importances']

                if isinstance(importances_df, pd.DataFrame):
                    # 获取特征名称和重要性值
                    feature_names = importances_df.index.tolist()
                    importances = importances_df.iloc[:, 0].values

                    all_importances.append(importances)
                    all_feature_names.append(feature_names)

                    # 提取项目名
                    project_name = filename.replace('.pkl', '')
                    file_info.append({
                        'filename': filename,
                        'project_name': project_name,
                        'n_features': len(feature_names),
                        'nonzero_features': np.sum(importances > 0)
                    })

                    print(f"{i:2d}. ✓ {filename}")
                else:
                    print(f"{i:2d}. ✗ {filename}: 数据格式不符")
            else:
                print(f"{i:2d}. ✗ {filename}: 没有特征重要性数据")

        except Exception as e:
            print(f"{i:2d}. ✗ {filename}: {str(e)[:50]}...")

    if not all_importances:
        print("没有成功加载数据")
        return None, None, None

    print(f"\n成功加载 {len(all_importances)} 个文件")
    return all_importances, all_feature_names, file_info


def create_final_chart():
    """创建最终的特征重要性图表"""
    print("\n" + "=" * 70)
    print("开始创建特征重要性图表")
    print("=" * 70)

    # 1. 设置中文字体
    print("\n1. 配置中文字体...")
    use_chinese, font_name = setup_chinese_font()

    # 2. 加载数据
    print("\n2. 加载数据...")
    all_importances, all_feature_names, file_info = load_all_pkl_data()

    if all_importances is None:
        print("数据加载失败")
        return

    n_files = len(all_importances)

    # 3. 准备数据
    print("\n3. 准备数据...")

    # 获取特征名称（假设所有文件的特征顺序相同）
    feature_names = all_feature_names[0]
    n_features = len(feature_names)

    # 创建重要性矩阵
    importance_matrix = np.zeros((n_files, n_features))
    for i in range(n_files):
        importance_matrix[i] = all_importances[i]

    # 计算平均重要性
    avg_importances = np.mean(importance_matrix, axis=0)

    # 计算每个特征的出现次数（重要性>0）
    occurrence_counts = np.sum(importance_matrix > 0, axis=0)

    # 4. 选择前20个最重要的特征
    top_n = 20
    sorted_indices = np.argsort(avg_importances)[::-1]  # 从高到低
    top_indices = sorted_indices[:top_n]

    top_features = [feature_names[i] for i in top_indices]
    top_importances = avg_importances[top_indices]
    top_occurrences = occurrence_counts[top_indices]

    # 5. 创建图表
    print("\n4. 创建图表...")

    fig, ax = plt.subplots(figsize=(16, 10))

    # 设置颜色
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))

    # 创建水平条形图
    y_pos = np.arange(top_n)
    bars = ax.barh(y_pos, top_importances, color=colors, height=0.8)

    # 添加误差条（标准差）
    std_values = np.std(importance_matrix[:, top_indices], axis=0)
    ax.errorbar(top_importances, y_pos,
                xerr=std_values,
                fmt='none', ecolor='gray', alpha=0.5, capsize=3)

    # 设置y轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=11)
    ax.invert_yaxis()  # 最重要的特征在顶部

    # 设置x轴
    max_importance = top_importances.max()
    ax.set_xlim(0, max_importance * 1.15)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')

    # 设置标题和标签（根据字体支持选择语言）
    if use_chinese:
        ax.set_title('特征重要性TOP20', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('平均特征重要性分数', fontsize=13, fontweight='bold')
    else:
        ax.set_title('Feature Importance TOP20', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Average Importance Score', fontsize=13, fontweight='bold')

    plt.tight_layout()

    # 6. 保存图表
    output_dir = './single_merged_bar/'
    os.makedirs(output_dir, exist_ok=True)

    if use_chinese:
        chart_name = '特征重要性TOP20.png'
    else:
        chart_name = 'feature_importance_top20_english.png'

    chart_path = os.path.join(output_dir, chart_name)
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ 图表已保存: {chart_path}")

    # 7. 保存数据
    print("\n5. 保存分析数据...")

    # 创建特征统计表
    feature_stats = []
    for i, feature in enumerate(feature_names):
        importances = importance_matrix[:, i]
        non_zero = importances[importances > 0]

        if len(non_zero) > 0:
            stats = {
                'Feature': feature,
                'Avg_Importance': avg_importances[i],
                'Std_Deviation': np.std(non_zero),
                'Occurrence': len(non_zero),
                'Occurrence_Percentage': f"{len(non_zero) / n_files:.1%}",
                'Max_Importance': np.max(non_zero),
                'Min_Importance': np.min(non_zero),
                'Total_Importance': np.sum(non_zero)
            }
            feature_stats.append(stats)

    # 转换为DataFrame并排序
    df_stats = pd.DataFrame(feature_stats)
    df_stats = df_stats.sort_values('Avg_Importance', ascending=False)

    # 保存数据
    csv_path = os.path.join(output_dir, 'feature_importance_statistics.csv')
    df_stats.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ 统计数据已保存: {csv_path}")

    # 保存Top 20数据
    top_20_path = os.path.join(output_dir, 'top_20_features.csv')
    df_stats.head(20).to_csv(top_20_path, index=False, encoding='utf-8')
    print(f"✓ Top 20数据已保存: {top_20_path}")

    # 生成分析报告
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("特征重要性分析报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n")
        f.write(f"字体状态: {'中文' if use_chinese else '英文'} ({font_name})\n")
        f.write(f"分析文件数: {n_files}\n")
        f.write(f"特征总数: {len(feature_names)}\n\n")

        f.write("Top 10 最重要的特征:\n")
        f.write("-" * 70 + "\n")
        for i, row in df_stats.head(10).iterrows():
            f.write(f"{i + 1:2d}. {row['Feature']:25s} | 平均重要性: {row['Avg_Importance']:.4f} | "
                    f"出现次数: {row['Occurrence']:2d}/{n_files} ({row['Occurrence_Percentage']})\n")

    print(f"✓ 分析报告已保存: {report_path}")

    # 8. 打印摘要
    print("\n" + "=" * 80)
    print("特征重要性分析摘要")
    print("=" * 80)
    print(f"字体状态: {'中文' if use_chinese else '英文'} ({font_name})")
    print(f"分析文件数: {n_files}")

    print(f"\nTop 10 最重要的特征:")
    print("-" * 80)
    print(f"{'排名':<4} {'特征':<25} {'平均重要性':<12} {'出现文件数':<12} {'标准差':<10}")
    print("-" * 80)

    for i, row in df_stats.head(10).iterrows():
        print(f"{i + 1:<4} {row['Feature']:<25} {row['Avg_Importance']:<12.4f} "
              f"{row['Occurrence']:<12} {row['Std_Deviation']:<10.4f}")

    print("=" * 80)


def main():
    """主函数"""
    print("=" * 80)
    print("软件缺陷预测特征重要性可视化工具")
    print("=" * 80)
    print("说明: 本工具将28个pkl文件的特征重要性合并分析")
    print("=" * 80)

    try:
        create_final_chart()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("程序执行完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()