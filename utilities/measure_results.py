import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# 设置中文字体 - 修复版
def set_chinese_font():
    """设置中文字体，确保中文能正常显示"""
    try:
        # 尝试直接设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("已设置中文字体")
    except:
        # 如果失败，尝试查找系统字体
        try:
            system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            chinese_fonts = []

            # 常见的中文字体名称
            common_chinese_fonts = [
                'SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong',
                'SimSun', 'STXihei', 'STKaiti', 'STSong', 'STFangsong'
            ]

            # 查找系统中是否存在这些字体
            for font in system_fonts:
                for chinese_font in common_chinese_fonts:
                    if chinese_font.lower() in font.lower():
                        chinese_fonts.append(font)
                        break

            if chinese_fonts:
                # 使用找到的第一个中文字体
                font_path = chinese_fonts[0]
                font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
                print(f"使用中文字体: {font_name}")
            else:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                print("使用默认字体")
        except:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            print("使用默认字体")

    plt.rcParams['axes.unicode_minus'] = False


# 调用字体设置函数
set_chinese_font()


class AllMetricsExtractor:
    """提取所有15个指标的类"""

    def __init__(self, base_path):
        self.base_path = base_path
        self.unsupervised_path = os.path.join(base_path, "allResults_unsupervised-methods")
        self.supervised_path = os.path.join(base_path, "supervised_classifiers_norandom_state")
        self.own_method_path = os.path.join(base_path, "all_result_INTC_K-means_label2.csv")

        # 定义15个指标的名称和顺序
        self.metrics = [
            "precision", "recall", "pf", "F1", "AUC",
            "g_measure", "g_mean", "bal", "MCC", "Popt",
            "Erecall", "Eprecision", "Efmeasure", "PMI", "IFA"
        ]

        # 存储所有方法的所有指标数据
        # 格式: metrics_data[metric_name][method_name] = [values]
        self.metrics_data = {metric: {} for metric in self.metrics}

        # 存储方法类型信息
        self.method_types = {}

    def extract_all_metrics(self):
        """从所有文件中提取所有15个指标"""
        print("=" * 80)
        print("开始从文件中提取所有15个指标")
        print("=" * 80)

        # 1. 处理无监督方法文件夹
        if os.path.exists(self.unsupervised_path):
            print(f"\n处理无监督方法文件夹: {self.unsupervised_path}")
            self._process_folder(self.unsupervised_path, "Unsupervised")
        else:
            print(f"警告: 无监督方法文件夹不存在: {self.unsupervised_path}")

        # 2. 处理监督方法文件夹
        if os.path.exists(self.supervised_path):
            print(f"\n处理监督方法文件夹: {self.supervised_path}")
            self._process_folder(self.supervised_path, "Supervised")
        else:
            print(f"警告: 监督方法文件夹不存在: {self.supervised_path}")

        # 3. 处理自己的方法文件
        if os.path.exists(self.own_method_path):
            print(f"\n处理自己的方法文件: {self.own_method_path}")
            self._process_own_method_file()
        else:
            print(f"警告: 自己的方法文件不存在: {self.own_method_path}")

        # 打印提取结果
        print(f"\n数据提取完成:")
        for metric in self.metrics:
            num_methods = len(self.metrics_data[metric])
            total_values = sum(len(values) for values in self.metrics_data[metric].values())
            print(f"  {metric}: {num_methods} 种方法, 共 {total_values} 个值")

    def _process_folder(self, folder_path, method_type):
        """处理单个文件夹中的所有文件"""
        # 查找所有CSV和Excel文件
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

        all_files = csv_files + excel_files

        if not all_files:
            print(f"  警告: 文件夹中没有CSV或Excel文件: {folder_path}")
            return

        print(f"  找到 {len(all_files)} 个文件")

        for file_name in all_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                # 读取文件
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:  # Excel文件
                    df = pd.read_excel(file_path)

                # 从文件名中提取方法名称（大写）
                method_name = self._extract_method_name(file_name).upper()

                print(f"\n    处理文件: {file_name}")
                print(f"    方法名称: {method_name}")
                print(f"    数据形状: {df.shape}")

                # 提取所有15个指标
                self._extract_all_columns(df, method_name, method_type, file_name)

            except Exception as e:
                print(f"    处理文件 {file_name} 时出错: {str(e)}")

    def _process_own_method_file(self):
        """处理自己的方法文件"""
        try:
            # 读取文件
            df = pd.read_csv(self.own_method_path)

            # 方法名称
            method_name = "INTC"  # 大写

            print(f"\n  处理自己的方法文件")
            print(f"  方法名称: {method_name}")
            print(f"  数据形状: {df.shape}")

            # 提取所有15个指标
            self._extract_all_columns(df, method_name, "Unsupervised (INTC)", "all_result_INTC_K-means_label2.csv")

        except Exception as e:
            print(f"  处理自己的方法文件时出错: {str(e)}")

    def _extract_method_name(self, file_name):
        """从文件名中提取方法名称"""
        # 移除文件扩展名
        name = file_name.rsplit('.', 1)[0]

        # 移除常见前缀
        prefixes = ['all_result_', 'result_', 'results_', 'output_']
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]

        # 特殊处理：将下划线替换为空格
        name = name.replace('_', ' ')

        return name

    def _extract_all_columns(self, df, method_name, method_type, file_name):
        """从DataFrame中提取所有15列指标"""
        # 存储方法类型
        self.method_types[method_name] = method_type

        # 检查列数
        num_columns = len(df.columns)
        print(f"      文件有 {num_columns} 列")

        # 提取前15列（如果存在）
        for i, metric in enumerate(self.metrics):
            if i < num_columns:
                column_data = df.iloc[:, i]
                values = self._clean_numeric_values(column_data, metric)

                if values:
                    # 存储到数据结构中
                    if method_name in self.metrics_data[metric]:
                        # 如果已经有这个方法的该指标数据，合并
                        self.metrics_data[metric][method_name].extend(values)
                    else:
                        self.metrics_data[metric][method_name] = values

                    print(f"      {metric}: 提取了 {len(values)} 个值")
                else:
                    print(f"      {metric}: 没有提取到有效值")
            else:
                print(f"      {metric}: 文件中没有第{i + 1}列")

    def _clean_numeric_values(self, column_data, metric_name):
        """清理数值数据"""
        values = []
        for idx, item in enumerate(column_data):
            try:
                # 如果是NaN，跳过
                if pd.isna(item):
                    continue

                # 尝试转换为浮点数
                if isinstance(item, (int, float, np.integer, np.floating)):
                    val = float(item)
                elif isinstance(item, str):
                    # 清理字符串
                    item_clean = item.strip()
                    # 移除可能的空白字符和特殊字符
                    item_clean = item_clean.replace(',', '.')  # 将逗号替换为点
                    # 移除百分号
                    if '%' in item_clean:
                        item_clean = item_clean.replace('%', '')
                        val = float(item_clean) / 100.0
                    else:
                        val = float(item_clean)
                else:
                    continue

                # 验证值范围（根据指标的不同）
                if metric_name in ["precision", "recall", "pf", "F1", "AUC", "g_measure", "g_mean", "bal", "MCC",
                                   "Popt", "Erecall", "Eprecision", "Efmeasure"]:
                    # 这些指标通常应在0-1范围内
                    if val < 0 or val > 1:
                        # 不打印警告，直接跳过
                        continue

                values.append(val)

            except Exception as e:
                # 跳过无法转换的值
                continue

        return values

    def save_all_metrics_to_excel(self):
        """将提取的所有指标保存到Excel文件"""
        print("\n" + "=" * 80)
        print("将所有指标保存到Excel文件")
        print("=" * 80)

        # 为每个指标创建一个Excel文件
        for metric in self.metrics:
            if self.metrics_data[metric]:
                metric_df = self._create_metric_dataframe(metric)
                excel_path = os.path.join(self.base_path, f"{metric}_all_methods.xlsx")
                metric_df.to_excel(excel_path, index=False)
                print(f"{metric}指标已保存到: {excel_path}")
            else:
                print(f"{metric}: 没有数据可保存")

        # 创建统计摘要文件
        self._create_statistics_summary()

    def _create_metric_dataframe(self, metric):
        """为单个指标创建数据框"""
        data_dict = {}

        # 获取该指标的所有方法数据
        method_data = self.metrics_data[metric]

        if not method_data:
            return pd.DataFrame()

        # 确定最大长度
        max_len = max(len(values) for values in method_data.values())

        # 填充数据使所有列表长度一致
        for method_name, values in method_data.items():
            padded_values = values + [np.nan] * (max_len - len(values))
            data_dict[method_name] = padded_values

        return pd.DataFrame(data_dict)

    def _create_statistics_summary(self):
        """创建统计摘要"""
        print("\n创建统计摘要...")

        stats_data = []

        for metric in self.metrics:
            method_data = self.metrics_data[metric]

            for method_name, values in method_data.items():
                if values:
                    method_type = self.method_types.get(method_name, "Unknown")

                    stats_data.append({
                        '指标': metric,
                        '方法名称': method_name,
                        '方法类型': method_type,
                        '样本数': len(values),
                        '均值': np.mean(values),
                        '标准差': np.std(values),
                        '中位数': np.median(values),
                        '最小值': np.min(values),
                        '最大值': np.max(values),
                        '下四分位': np.percentile(values, 25) if len(values) >= 4 else np.nan,
                        '上四分位': np.percentile(values, 75) if len(values) >= 4 else np.nan
                    })

        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_path = os.path.join(self.base_path, "all_metrics_statistics.xlsx")
            stats_df.to_excel(stats_path, index=False)
            print(f"统计摘要已保存到: {stats_path}")

    def _get_display_metric_name(self, metric):
        """获取指标的显示名称"""
        # 对特定指标进行特殊处理
        if metric == "g_mean":
            return "G-mean"
        # 对其他指标使用大写
        return metric.upper()

    def visualize_all_metrics(self):
        """可视化所有15个指标的箱线图"""
        print("\n" + "=" * 80)
        print("可视化所有指标的箱线图")
        print("=" * 80)

        # 为每个指标创建箱线图
        for metric in self.metrics:
            self._create_metric_boxplot(metric)

        # 创建综合对比图
        self._create_combined_visualization()

    def _create_metric_boxplot(self, metric):
        """为单个指标创建箱线图"""
        method_data = self.metrics_data[metric]

        if not method_data:
            print(f"{metric}: 没有数据可用于可视化")
            return

        # 准备数据
        data_to_plot = []
        labels = []
        colors = []

        # 定义颜色映射
        color_map = {
            "Unsupervised": "lightblue",
            "Supervised": "lightgreen",
            "Unsupervised (INTC)": "orange"  # 自己的方法用橙色
        }

        for method_name, values in method_data.items():
            if values:
                data_to_plot.append(values)
                # 修改特定方法名称的显示
                if method_name == "K-MEDOIDS":
                    display_name = "K-medoids"
                elif method_name == "XGBOOST":
                    display_name = "XGBoost"
                else:
                    display_name = method_name  # 其他保持不变
                labels.append(display_name)

                # 获取方法类型并确定颜色
                method_type = self.method_types.get(method_name, "Unknown")
                colors.append(color_map.get(method_type, "gray"))

        if not data_to_plot:
            return

        # 创建图形
        plt.figure(figsize=(max(14, len(data_to_plot) * 0.7), 8))

        # 创建箱线图
        boxplot = plt.boxplot(data_to_plot, patch_artist=True,
                              labels=labels, showmeans=True, meanline=True)

        # 设置颜色
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 设置中位线颜色
        for median in boxplot['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        # 设置均值线颜色
        for mean in boxplot['means']:
            mean.set_color('blue')
            mean.set_linewidth(2)

        # 获取指标的显示名称
        display_metric = self._get_display_metric_name(metric)

        # 设置标题和标签 - 使用中文
        plt.title(f'{display_metric} 指标对比', fontsize=16, fontweight='bold')
        plt.ylabel(f'{display_metric} 值', fontsize=12)
        plt.xlabel('方法名称', fontsize=12)

        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')

        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 添加图例
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='无监督方法'),
            Patch(facecolor='lightgreen', alpha=0.7, label='监督方法'),
            Patch(facecolor='orange', alpha=0.7, label='INTC方法(ours)'),
            # Line2D([0], [0], color='red', linewidth=2, label='中位数'),
            # Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='均值')
        ]
        plt.legend(handles=legend_elements, loc='upper left')

        # 添加说明文本
        plt.figtext(0.02, 0.02,
                    '说明：红色实线表示中位数，蓝色虚线表示均值',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.base_path, f"{metric}_boxplot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{metric}箱线图已保存到: {save_path}")

        plt.show()

    def _create_combined_visualization(self):
        """创建综合可视化（按方法类型分组）"""
        print("\n创建综合可视化...")

        # 选择一些关键指标进行综合可视化
        key_metrics = ["precision", "recall", "F1", "AUC", "MCC", "g_mean"]

        # 创建图形
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, metric in enumerate(key_metrics):
            if idx >= len(axes):
                break

            ax = axes[idx]
            method_data = self.metrics_data[metric]

            if not method_data:
                # 获取指标的显示名称
                display_metric = self._get_display_metric_name(metric)
                ax.text(0.5, 0.5, f"没有{display_metric}数据",
                        ha='center', va='center', fontsize=12)
                ax.set_title(f'{display_metric}', fontsize=14)
                continue

            # 按方法类型分组数据
            unsupervised_data = []
            unsupervised_labels = []
            supervised_data = []
            supervised_labels = []
            own_data = []
            own_label = []

            for method_name, values in method_data.items():
                if not values:
                    continue

                method_type = self.method_types.get(method_name, "Unknown")

                if method_type == "Unsupervised (INTC)":
                    own_data.append(values)
                    own_label.append(method_name)
                elif method_type == "Unsupervised":
                    unsupervised_data.append(values)
                    # 修改特定方法名称的显示
                    if method_name == "K-MEDOIDS":
                        display_name = "K-medoids"
                    else:
                        display_name = method_name
                    unsupervised_labels.append(display_name)
                elif method_type == "Supervised":
                    supervised_data.append(values)
                    # 修改特定方法名称的显示
                    if method_name == "XGBOOST":
                        display_name = "XGBoost"
                    else:
                        display_name = method_name
                    supervised_labels.append(display_name)

            # 创建分组箱线图
            all_data = []
            all_labels = []

            if unsupervised_data:
                all_data.extend(unsupervised_data)
                all_labels.extend(unsupervised_labels)

            if supervised_data:
                all_data.extend(supervised_data)
                all_labels.extend(supervised_labels)

            if own_data:
                all_data.extend(own_data)
                all_labels.extend(own_label)

            if all_data:
                boxplot = ax.boxplot(all_data, patch_artist=True,
                                     labels=all_labels, showmeans=True)

                # 设置颜色
                colors = []
                for method_name in all_labels:
                    # 获取原始方法名称
                    if method_name == "K-medoids":
                        original_name = "K-MEDOIDS"
                    elif method_name == "XGBoost":
                        original_name = "XGBOOST"
                    else:
                        original_name = method_name

                    method_type = self.method_types.get(original_name, "Unknown")
                    if method_type == "Unsupervised (INTC)":
                        colors.append('orange')
                    elif method_type == "Unsupervised":
                        colors.append('lightblue')
                    elif method_type == "Supervised":
                        colors.append('lightgreen')
                    else:
                        colors.append('gray')

                for patch, color in zip(boxplot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # 设置中位线颜色
                for median in boxplot['medians']:
                    median.set_color('red')
                    median.set_linewidth(1.5)

                # 获取指标的显示名称
                display_metric = self._get_display_metric_name(metric)

                ax.set_title(f'{display_metric}', fontsize=14, fontweight='bold')
                ax.set_ylabel(f'{display_metric} 值', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            else:
                # 获取指标的显示名称
                display_metric = self._get_display_metric_name(metric)
                ax.text(0.5, 0.5, f"没有{display_metric}数据",
                        ha='center', va='center', fontsize=12)
                ax.set_title(f'{display_metric}', fontsize=14)

        plt.suptitle('关键指标综合对比', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()

        # 保存图像
        save_path = os.path.join(self.base_path, "key_metrics_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"关键指标综合对比图已保存到: {save_path}")

        plt.show()

    def analyze_performance(self):
        """分析INTC(OWNS)方法在所有指标上的表现"""
        print("\n" + "=" * 80)
        print("INTC方法性能分析")
        print("=" * 80)

        own_method = "INTC"

        for metric in self.metrics:
            if own_method in self.metrics_data[metric]:
                values = self.metrics_data[metric][own_method]
                if values:
                    mean_val = np.mean(values)
                    display_metric = self._get_display_metric_name(metric)
                    print(f"{display_metric}: 均值 = {mean_val:.4f}, 样本数 = {len(values)}")

        # 与其他方法的对比
        print("\n与其他方法的对比:")
        for metric in ["AUC", "F1", "MCC", "precision", "recall"]:
            if own_method in self.metrics_data[metric]:
                own_values = self.metrics_data[metric][own_method]
                if own_values:
                    own_mean = np.mean(own_values)

                    # 计算所有方法的平均值
                    all_means = []
                    for method_name, values in self.metrics_data[metric].items():
                        if values:
                            all_means.append(np.mean(values))

                    if all_means:
                        avg_all = np.mean(all_means)
                        display_metric = self._get_display_metric_name(metric)
                        print(
                            f"{display_metric}: INTC = {own_mean:.4f}, 所有方法平均 = {avg_all:.4f}, 差异 = {own_mean - avg_all:.4f}")


def main():
    """主函数"""
    print("=" * 80)
    print("软件缺陷预测方法综合性能分析工具")
    print("提取15个指标进行对比分析")
    print("=" * 80)

    # 设置基础路径
    base_path = "F:/interpretability/code/INTC/all-results"

    print(f"基础路径: {base_path}")
    print(f"无监督方法路径: {os.path.join(base_path, 'allResults_unsupervised-methods')}")
    print(f"监督方法路径: {os.path.join(base_path, 'supervised_classifiers_norandom_state')}")
    print(f"INTC方法文件: {os.path.join(base_path, 'all_result_INTC_K-means_label2.csv')}")

    # 检查路径是否存在
    required_paths = [
        os.path.join(base_path, "allResults_unsupervised-methods"),
        os.path.join(base_path, "supervised_classifiers_norandom_state"),
        os.path.join(base_path, "all_result_INTC_K-means_label2.csv")
    ]

    for path in required_paths:
        if not os.path.exists(path):
            print(f"警告: 路径不存在: {path}")

    # 初始化提取器
    extractor = AllMetricsExtractor(base_path)

    # 提取所有指标
    extractor.extract_all_metrics()

    # 保存所有指标到Excel
    extractor.save_all_metrics_to_excel()

    # 可视化所有指标
    extractor.visualize_all_metrics()

    # 分析INTC(OWNS)方法性能
    extractor.analyze_performance()

    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)

    # 打印生成的输出文件
    print("\n生成的输出文件:")
    print("1. 15个指标文件: precision_all_methods.xlsx, recall_all_methods.xlsx, ...")
    print("2. 统计摘要文件: all_metrics_statistics.xlsx")
    print("3. 15个箱线图: precision_boxplot.png, recall_boxplot.png, ...")
    print("4. 关键指标综合对比图: key_metrics_comparison.png")


if __name__ == "__main__":
    # 检查依赖包
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        main()
    except ImportError as e:
        print(f"缺少依赖包: {e}")
        print("\n请安装以下依赖包:")
        print("conda activate intc")
        print("pip install pandas numpy matplotlib scipy openpyxl")