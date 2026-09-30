import pandas as pd
import os
import glob
import warnings

warnings.filterwarnings('ignore')


def integrate_performance_tables(input_folder, output_folder):



    os.makedirs(output_folder, exist_ok=True)
    print(f": {output_folder}")


    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    csv_files = [f for f in csv_files if not os.path.basename(f).lower() == 'icusdp.csv']

    if len(csv_files) == 0:
        return

    print(f"✅ {len(csv_files)} ...")
    for idx, f in enumerate(csv_files, 1):
        print(f"   {idx:2d}. {os.path.basename(f)}")


    all_metric_data = {}
    project_list = None

    for file_path in csv_files:

        file_name = os.path.basename(file_path)
        file_prefix = "result_"
        file_suffix = ".csv"

        if file_name.startswith(file_prefix) and file_name.endswith(file_suffix):
            method_name = file_name[len(file_prefix):-len(file_suffix)]
        else:

            method_name = os.path.splitext(file_name)[0]
        print(f"\n: {method_name}")


        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            print(f"   sucession: {df.shape}（: {len(df)}, : {len(df.columns)-1}）")
        except Exception as e:
            print(f"   ❌  {file_name}: {str(e)[:50]}，")
            continue


        current_projects = df.iloc[:, 0].tolist()
        if project_list is None:
            project_list = current_projects

            for metric in df.columns[1:]:
                all_metric_data[metric] = {'Project': project_list}
        else:

            if len(current_projects) != len(project_list):
                print(f"   （{len(current_projects)}，{len(project_list)}）")
                continue


        metrics = df.columns[1:]
        for metric in metrics:
            if metric not in all_metric_data:

                all_metric_data[metric] = {'Project': project_list}

            all_metric_data[metric][method_name] = df[metric].tolist()


    if not all_metric_data:
        return

    metric_count = len(all_metric_data)
    print(f"\n {metric_count} ")

    for idx, (metric_name, metric_data) in enumerate(all_metric_data.items(), 1):

        metric_df = pd.DataFrame(metric_data)
        metric_df.set_index('Project', inplace=True)


        output_file = os.path.join(output_folder, f"{metric_name}.csv")
        metric_df.to_csv(output_file, encoding='utf-8-sig')


        method_count = len(metric_df.columns)
        print(f"   {idx:2d}/{metric_count} 生成 {metric_name}.csv（包含{method_count}个方法）")





if __name__ == "__main__":

    INPUT_FOLDER = r"F:\ICUSDP\INTC\INTC\result_20251016\all"

    OUTPUT_FOLDER = r"F:\ICUSDP\INTC\INTC\result_20251016\performance comparision"


    print("=" * 70)
    print("=" * 70)
    print(f": {INPUT_FOLDER}")
    print(f": {OUTPUT_FOLDER}")
    print("=" * 70)


    if not os.path.exists(INPUT_FOLDER):
        print(f"❌：{INPUT_FOLDER}")

    else:

        integrate_performance_tables(INPUT_FOLDER, OUTPUT_FOLDER)
