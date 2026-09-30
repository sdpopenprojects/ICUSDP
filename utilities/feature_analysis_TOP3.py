from collections import Counter
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
  plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
  plt.rcParams['axes.unicode_minus'] = False
  plt.rcParams['figure.dpi'] = 150


  input_dir = r'E:/ICUSDP/INTC/ICUSDP-main/new/FS/RQ4/topk/ICUSDP'
  output_dir = r'E:/ICUSDP/INTC/ICUSDP-main/new/FS/RQ3/feature_analysis_results_top3_groups'
  os.makedirs(output_dir, exist_ok=True)

  top3_file = os.path.join(input_dir, 'top3_v1.csv')
  if not os.path.exists(top3_file):
    print(f'[ERROR] : {top3_file}')
    return

  print(f': {top3_file}')
  top3_df = pd.read_csv(top3_file, header=None)
  total_projects = len(top3_df)


  project_names = [
      'activemq-5.0.0',
      'activemq-5.1.0',
      'activemq-5.2.0',
      'activemq-5.3.0',
      'activemq-5.8.0',
      'derby-10.2.1.6',
      'derby-10.3.1.4',
      'derby-10.5.1.1',
      'groovy-1_5_7',
      'groovy-1_6_beta1',
      'groovy-1_6_beta2',
      'hbase-0.94.0',
      'hbase-0.95.0',
      'hbase-0.95.2',
      'hive-0.10.0',
      'hive-0.12.0',
      'hive-0.9.0',
      'jruby-1.1',
      'jruby-1.4.0',
      'jruby-1.5.0',
      'jruby-1.7.0',
      'lucene-2.3.0',
      'lucene-2.9.0',
      'lucene-3.0.0',
      'lucene-3.1',
      'wicket-1.3.0-beta2',
      'wicket-1.3.0-beta1',
      'wicket-1.5.3',
  ]

  if len(project_names) > total_projects:
    project_names = project_names[:total_projects]


  feature_project_counts = {}
  matrix_data = []

  for idx, row in top3_df.iterrows():
    feats = [f.strip() for f in str(row[0]).split(',') if f.strip()]
    matrix_data.append(feats)
    for f in set(feats):
      feature_project_counts[f] = feature_project_counts.get(f, 0) + 1


  df_stab = pd.DataFrame([
      {
          'Feature': k,
          'Projects_Count': v,
          'Percentage': (v / total_projects) * 100,
      }
      for k, v in feature_project_counts.items()
  ]).sort_values(by='Percentage', ascending=False)


  stats_csv_path = os.path.join(output_dir, 'global_top3_selection_statistics.csv')
  df_stab.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
  print(f'[INFO] : {stats_csv_path}')


  df_stab_sorted_for_plot = df_stab.sort_values(by='Percentage', ascending=True)

  plt.figure(figsize=(14, 10))
  bars = plt.barh(
      df_stab_sorted_for_plot['Feature'],
      df_stab_sorted_for_plot['Percentage'],
      color='steelblue',
      edgecolor='black',
      height=0.75,
  )


  plt.title(
      'Global Feature Stability & Selection Percentage (Top-3)',
      fontsize=26,
      fontweight='bold',
      pad=20,
  )
  plt.xlim(0, 115)
  plt.xticks(fontsize=22)
  plt.yticks(fontsize=22, fontweight='bold')
  plt.grid(axis='x', linestyle='--', alpha=0.7)


  for bar, value in zip(bars, df_stab_sorted_for_plot['Percentage']):
    plt.text(
        value + 1.5,
        bar.get_y() + bar.get_height() / 2,
        f'{value:.1f}%',
        va='center',
        ha='left',
        fontsize=20,
        fontweight='bold',
    )

  plt.tight_layout()
  bar_png = os.path.join(output_dir, 'global_top3_percentage_bar.png')
  bar_pdf = os.path.join(output_dir, 'global_top3_percentage_bar.pdf')
  plt.savefig(bar_png, dpi=300, bbox_inches='tight')
  plt.savefig(bar_pdf, format='pdf', bbox_inches='tight')
  plt.close()
  print(f'[INFO] : {bar_png}')


  unique_features = df_stab['Feature'].tolist()
  heatmap_matrix = pd.DataFrame(
      0, index=project_names, columns=unique_features
  )

  for i, feats in enumerate(matrix_data):
    if i < len(project_names):
      p_name = project_names[i]
      for f in feats:
        if f in heatmap_matrix.columns:
          heatmap_matrix.loc[p_name, f] = 1

  heatmap_csv_path = os.path.join(output_dir, 'global_top3_heatmap_matrix.csv')
  heatmap_matrix.to_csv(heatmap_csv_path, encoding='utf-8-sig')
  print(f'[INFO] : {heatmap_csv_path}')


  plt.figure(figsize=(18, max(12, len(project_names) * 0.45)))


  ax = sns.heatmap(
      heatmap_matrix,
      cmap='Blues',
      linewidths=0.5,
      linecolor='#dcdcdc',
      vmin=0,
      vmax=1,
      cbar_kws={'label': 'Present in Top-3 (1=Yes, 0=No)'},
  )


  plt.xticks(
      range(len(unique_features)),
      unique_features,
      rotation=45,
      ha='right',
      fontsize=22,
      fontweight='bold',
  )
  plt.yticks(
      range(len(project_names)),
      project_names,
      fontsize=20,
      fontweight='bold',
      rotation=0,
  )


  cbar = ax.collections[0].colorbar
  cbar.set_label(
      'Present in Top-3 (1=Yes, 0=No)', fontsize=20, fontweight='bold'
  )
  cbar.ax.tick_params(labelsize=18)


  plt.title(
      'Cross-Project Feature Stability Heatmap (Top-3)',
      fontsize=28,
      fontweight='bold',
      pad=25,
  )

  plt.tight_layout()

  heat_png = os.path.join(output_dir, 'global_top3_stability_heatmap.png')
  heat_pdf = os.path.join(output_dir, 'global_top3_stability_heatmap.pdf')
  plt.savefig(heat_png, dpi=300, bbox_inches='tight')
  plt.savefig(heat_pdf, format='pdf', bbox_inches='tight')
  plt.close()
  print(f'[INFO] : {heat_png}')
  print(
      f'\n[SUCCESS] :\n{output_dir}'
  )


if __name__ == '__main__':
  main()