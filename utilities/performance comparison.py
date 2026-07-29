import pandas as pd
import os
import glob
import warnings

warnings.filterwarnings('ignore')


def integrate_performance_tables(input_folder, output_folder):
    """
    整合15个方法表格，按指标分类生成对比表（仅CSV）
    每个对比表包含：1个指标 + 所有方法在各项目上的结果
    """

    # 1. 创建输出文件夹（自动创建，无需手动建）
    os.makedirs(output_folder, exist_ok=True)
    print(f"✅ 输出文件夹已准备: {output_folder}")

    # 2. 获取输入文件夹中的所有CSV文件（匹配您的15个表格）
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    # 过滤非结果文件（如ICUSDP.csv若为原始数据，可跳过；若为结果则保留）
    # 若ICUSDP.csv是结果文件，可删除下方过滤代码
    csv_files = [f for f in csv_files if not os.path.basename(f).lower() == 'icusdp.csv']

    if len(csv_files) == 0:
        print("❌ 错误：在输入文件夹中未找到有效CSV结果文件！")
        return

    print(f"✅ 找到 {len(csv_files)} 个方法结果文件，开始处理...")
    for idx, f in enumerate(csv_files, 1):
        print(f"   {idx:2d}. {os.path.basename(f)}")

    # 3. 读取所有文件，提取项目名、方法名、指标数据
    all_metric_data = {}  # 存储结构：{指标名: {项目列: [项目1,项目2...], 方法1: [值1,值2...], 方法2: [值1,值2...]}}
    project_list = None    # 统一存储项目名称（从第一个文件提取，确保所有文件项目顺序一致）

    for file_path in csv_files:
        # 3.1 提取方法名（从文件名解析，如"result_CLA.csv"→"CLA"，"result_TCLP.csv"→"TCLP"）
        file_name = os.path.basename(file_path)
        file_prefix = "result_"
        file_suffix = ".csv"
        # 截取"result_"和".csv"之间的部分作为方法名
        if file_name.startswith(file_prefix) and file_name.endswith(file_suffix):
            method_name = file_name[len(file_prefix):-len(file_suffix)]
        else:
            # 若文件名不匹配标准格式，用完整文件名（去除后缀）作为方法名
            method_name = os.path.splitext(file_name)[0]
        print(f"\n🔍 处理方法: {method_name}")

        # 3.2 读取CSV文件（假设第一列是项目名，其余列为指标）
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            print(f"   ✅ 成功读取，数据形状: {df.shape}（项目数: {len(df)}, 指标数: {len(df.columns)-1}）")
        except Exception as e:
            print(f"   ❌ 读取失败 {file_name}: {str(e)[:50]}，跳过该文件")
            continue

        # 3.3 统一项目名称（从第一个文件提取项目列，后续文件按此对齐）
        current_projects = df.iloc[:, 0].tolist()  # 第一列是项目名（如equinox、jdt等）
        if project_list is None:
            project_list = current_projects
            # 初始化所有指标的存储结构（添加项目列）
            for metric in df.columns[1:]:  # 跳过第一列（项目名），其余是指标
                all_metric_data[metric] = {'Project': project_list}
        else:
            # 检查当前文件项目数与第一个文件是否一致（避免数据错位）
            if len(current_projects) != len(project_list):
                print(f"   ⚠️ 项目数不匹配（当前{len(current_projects)}个，标准{len(project_list)}个），跳过该文件")
                continue

        # 3.4 提取当前方法的所有指标数据，存入对应指标的字典
        metrics = df.columns[1:]  # 所有指标列（如precision、recall、AUC等）
        for metric in metrics:
            if metric not in all_metric_data:
                # 若当前指标是新指标（第一个文件未包含），补充初始化
                all_metric_data[metric] = {'Project': project_list}
            # 提取当前方法在该指标下的所有项目值，存入对应指标
            all_metric_data[metric][method_name] = df[metric].tolist()

    # 4. 按指标生成对比表并保存到输出文件夹
    if not all_metric_data:
        print("\n❌ 无有效指标数据，无法生成对比表！")
        return

    metric_count = len(all_metric_data)
    print(f"\n📊 共提取到 {metric_count} 个指标，开始生成对比表...")

    for idx, (metric_name, metric_data) in enumerate(all_metric_data.items(), 1):
        # 4.1 构建对比表DataFrame（项目列为索引，方法列为数据列）
        metric_df = pd.DataFrame(metric_data)
        metric_df.set_index('Project', inplace=True)  # 项目名设为行索引，更易读

        # 4.2 保存为CSV文件（文件名：指标名.csv，如precision.csv、AUC.csv）
        output_file = os.path.join(output_folder, f"{metric_name}.csv")
        metric_df.to_csv(output_file, encoding='utf-8-sig')

        # 4.3 打印进度
        method_count = len(metric_df.columns)
        print(f"   {idx:2d}/{metric_count} 生成 {metric_name}.csv（包含{method_count}个方法）")

    # 5. 输出最终统计信息
    print(f"\n🎉 所有对比表生成完成！")
    print(f"📁 输出位置: {output_folder}")
    print(f"📋 统计：{metric_count} 个指标对比表，每个表包含 {len(csv_files)} 个方法的结果")


# ------------------------------------------------------------------------------
# 主程序入口（路径已按您的需求设置，直接运行）
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 您的输入路径（15个表格所在文件夹）
    INPUT_FOLDER = r"F:\ICUSDP\INTC\INTC\result_20251016\all"
    # 您的输出路径（按指标分类的对比表保存文件夹）
    OUTPUT_FOLDER = r"F:\ICUSDP\INTC\INTC\result_20251016\performance comparision"

    # 打印程序信息
    print("=" * 70)
    print("              15个方法结果表格整合程序（按指标分类）")
    print("=" * 70)
    print(f"📥 输入文件夹: {INPUT_FOLDER}")
    print(f"📤 输出文件夹: {OUTPUT_FOLDER}")
    print("📋 功能：每个指标生成1个对比表，包含所有方法在各项目的结果")
    print("=" * 70)

    # 检查输入文件夹是否存在
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ 错误：输入文件夹不存在！路径：{INPUT_FOLDER}")
        print("   请检查路径是否正确（如文件夹名、盘符是否正确）")
    else:
        # 执行整合逻辑
        integrate_performance_tables(INPUT_FOLDER, OUTPUT_FOLDER)
        print("\n✅ 程序执行完毕！")