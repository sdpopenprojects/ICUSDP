import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, kendalltau


plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


method_names = ['ICUSDP', 'DT', 'RF', 'GBM', 'XGBoost', 'LR', 'linearSVM']
base_dir = r'E:\ICUSDP\INTC\ICUSDP-main\new\FS\RQ4\topk'
icusdp_name = 'ICUSDP'
contrast_methods = [m for m in method_names if m != icusdp_name]

output_dir = os.path.join(base_dir, 'summary_results')
os.makedirs(output_dir, exist_ok=True)


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
    'wicket-1.3.0-beta1',
    'wicket-1.3.0-beta2',
    'wicket-1.5.3',
]



def read_topk_csv(file_path):
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = [line.strip().replace('"', '') for line in f.readlines()]
  feature_sets = []
  for line in lines:
    if line:
      feats = set([x.strip() for x in line.split(',') if x.strip()])
      feature_sets.append(feats)
    else:
      feature_sets.append(set())
  return feature_sets


def calc_overlap(set_a, set_b):
  union_len = len(set_a.union(set_b))
  if union_len == 0:
    return 0.0
  return len(set_a.intersection(set_b)) / union_len


def get_tau_agreement(val):
  abs_val = abs(val)
  if abs_val <= 0.3:
    return 'weak'
  elif abs_val <= 0.6:
    return 'moderate'
  else:
    return 'strong'


top1_data = {}
top3_data = {}
ar_data = {}
ar_global_data = {}

for m in method_names:
  t1_path = os.path.join(base_dir, m, 'top1_v1.csv')
  t3_path = os.path.join(base_dir, m, 'top3_v1.csv')
  ar_path = os.path.join(base_dir, m, 'npskesd_v1.csv')
  ar_global_path = os.path.join(
      base_dir, m, 'npskesd_AR_v1.csv'
  )

  top1_data[m] = read_topk_csv(t1_path)
  top3_data[m] = read_topk_csv(t3_path)


  df_ar = pd.read_csv(ar_path)
  if 'Unnamed: 0' in df_ar.columns:
    df_ar = df_ar.drop(columns=['Unnamed: 0'])
  ar_data[m] = df_ar


  df_ar_global = pd.read_csv(ar_global_path)
  if 'Unnamed: 0' in df_ar_global.columns:
    df_ar_global = df_ar_global.drop(columns=['Unnamed: 0'])
  ar_global_data[m] = df_ar_global

record_rows = []
for proj_idx, p_name in enumerate(project_names):
  for m in contrast_methods:
    set_icu_1 = (
        top1_data[icusdp_name][proj_idx]
        if proj_idx < len(top1_data[icusdp_name])
        else set()
    )
    set_m_1 = (
        top1_data[m][proj_idx] if proj_idx < len(top1_data[m]) else set()
    )
    t1_ov = calc_overlap(set_icu_1, set_m_1)

    set_icu_3 = (
        top3_data[icusdp_name][proj_idx]
        if proj_idx < len(top3_data[icusdp_name])
        else set()
    )
    set_m_3 = top3_data[m][proj_idx] if proj_idx < len(top3_data[m]) else set()
    t3_ov = calc_overlap(set_icu_3, set_m_3)


    icu_ar = ar_data[icusdp_name].iloc[proj_idx]
    m_ar = ar_data[m].iloc[proj_idx].reindex(icu_ar.index)


    tau, p_val = kendalltau(icu_ar.values, m_ar.values)

    record_rows.append({
        'Project': p_name,
        'Method': m,
        'Top1_Overlap': t1_ov,
        'Top3_Overlap': t3_ov,
        'Kendall_Tau': tau if not np.isnan(tau) else 0.0,
        'p_value': p_val if not np.isnan(p_val) else 1.0,
    })

df_project_level = pd.DataFrame(record_rows)
detail_csv_path = os.path.join(output_dir, 'Project_Level_Interpretability.csv')
df_project_level.to_csv(detail_csv_path, index=False, encoding='utf-8-sig')


df_mean = df_project_level.groupby('Method')[
    ['Top1_Overlap', 'Top3_Overlap', 'Kendall_Tau', 'p_value']
].mean().reindex(contrast_methods)


def get_top1_agreement(val):
  return 'high' if val > 0.5 else 'low'


def get_top3_agreement(val):
  if val <= 0.25:
    return 'negligible'
  elif val <= 0.5:
    return 'small'
  elif val <= 0.75:
    return 'medium'
  else:
    return 'large'


df_topk_summary = pd.DataFrame({
    'Contrast_Method': contrast_methods,
    'Top-1 Overlap': df_mean['Top1_Overlap'].values,
    'Top-1 Agreement': [
        get_top1_agreement(v) for v in df_mean['Top1_Overlap'].values
    ],
    'Top-3 Overlap': df_mean['Top3_Overlap'].values,
    'Top-3 Agreement': [
        get_top3_agreement(v) for v in df_mean['Top3_Overlap'].values
    ],
})

df_tau_summary = pd.DataFrame({
    'Contrast_Method': contrast_methods,
    "Kendall's Tau": df_mean['Kendall_Tau'].values,
    'p-value': df_mean['p_value'].values,
    'Tau Agreement': [
        get_tau_agreement(v) for v in df_mean['Kendall_Tau'].values
    ],
})

topk_csv_path = os.path.join(output_dir, 'Topk_Overlap_Summary.csv')
tau_csv_path = os.path.join(output_dir, 'Kendalls_Tau_Summary.csv')
df_topk_summary.to_csv(topk_csv_path, index=False, encoding='utf-8-sig')
df_tau_summary.to_csv(tau_csv_path, index=False, encoding='utf-8-sig')



global_tau_rows = []
icu_global_ar = ar_global_data[icusdp_name].iloc[0]

for m in contrast_methods:
  m_global_ar = ar_global_data[m].iloc[0].reindex(icu_global_ar.index)
  g_tau, g_pval = kendalltau(icu_global_ar.values, m_global_ar.values)
  global_tau_rows.append({
      'Contrast_Method': m,
      "Kendall's Tau": g_tau if not np.isnan(g_tau) else 0.0,
      'p-value': g_pval if not np.isnan(g_pval) else 1.0,
      'Tau Agreement': get_tau_agreement(g_tau if not np.isnan(g_tau) else 0.0),
  })

df_global_tau_summary = pd.DataFrame(global_tau_rows)
global_tau_csv_path = os.path.join(
    output_dir, 'Global_Kendalls_Tau_Summary.csv'
)
df_global_tau_summary.to_csv(
    global_tau_csv_path, index=False, encoding='utf-8-sig'
)

print('\n=== Top-k Overlap  ===')
print(df_topk_summary.to_string(index=False))

print("\n===  Kendall's Tau  ===")
print(df_tau_summary.to_string(index=False))

print('\n===  Kendall\'s Tau ')
print(df_global_tau_summary.to_string(index=False))

print(
    f'\n✅ save:\n'
    f'   - {topk_csv_path}\n'
    f'   - {tau_csv_path}\n'
    f'   - {global_tau_csv_path}'
)


fig, axes = plt.subplots(
    1, 3, figsize=(24, 15), sharey=True, constrained_layout=False
)
metrics = [
    ('Top1_Overlap', 'Top-1 Overlap\n(High, Low)', [-0.1, 1.1]),
    ('Top3_Overlap', 'Top-3 Overlap\n(Negligible, Small, Medium, Large)', [
        -0.1,
        1.1,
    ]),
    ('Kendall_Tau', "Kendall's Tau\n(Weak, Moderate, Strong)", [-1.1, 1.1]),
]

colors = plt.cm.tab10(np.linspace(0, 1, len(contrast_methods)))
markers = ['o', '^', 's', 'P', 'X', 'd']
num_proj = len(project_names)

for ax_idx, (col_name, title_str, x_lim) in enumerate(metrics):
  ax = axes[ax_idx]

  all_values = df_project_level[col_name].values
  try:
    kde = gaussian_kde(all_values)
    # x_range = np.linspace(x_lim[0], x_lim[1], 200)
    x_range = np.linspace(x_lim[0], x_lim[1], 300)
    y_density = kde(x_range)
    y_density = y_density / y_density.max() * 0.6
  except:
    y_density = np.zeros(200)
    x_range = np.linspace(x_lim[0], x_lim[1], 200)

  for i, proj in enumerate(project_names):
    sub_df = df_project_level[df_project_level['Project'] == proj]

    ax.plot(
        x_range,
        i + y_density * 0.8,
        color='black',
        linewidth=1.1,
        alpha=0.7,
        zorder=1,
    )
    ax.fill_between(
        x_range,
        i,
        i + y_density * 0.8,
        color='lightgray',
        alpha=0.4,
        zorder=1,
    )
    ax.axhline(y=i, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    for j, m in enumerate(contrast_methods):
      val_row = sub_df[sub_df['Method'] == m][col_name]
      if not val_row.empty:
        val = val_row.values[0]
        ax.scatter(
            val,
            i,
            color=colors[j],
            marker=markers[j % len(markers)],
            s=90,
            edgecolors='black',
            linewidths=0.7,
            zorder=3,
        )


  ax.set_title(title_str, fontsize=22, fontweight='bold', pad=12)
  ax.set_xlim(x_lim)
  ax.set_yticks(range(num_proj))
  ax.set_yticklabels(project_names, fontsize=22, fontweight='semibold')
  ax.tick_params(axis='x', labelsize=22)
  for label in ax.get_xticklabels():
    label.set_fontweight('bold')
  ax.grid(axis='x', linestyle=':', alpha=0.6)

  if ax_idx == 0:
    ax.axvline(x=0.5, color='black', linestyle=':', alpha=0.6, linewidth=1.2)
  elif ax_idx == 1:
    for th in [0.25, 0.5, 0.75]:
      ax.axvline(x=th, color='black', linestyle=':', alpha=0.6, linewidth=1.2)
  elif ax_idx == 2:
    for th in [0.3, 0.6]:
      ax.axvline(x=th, color='black', linestyle=':', alpha=0.6, linewidth=1.2)


legend_handles = []
for j, m in enumerate(contrast_methods):
  handle = Line2D(
      [0],
      [0],
      marker=markers[j % len(markers)],
      color='w',
      markerfacecolor=colors[j],
      markeredgecolor='black',
      markersize=22,
      label=m,
  )
  legend_handles.append(handle)

fig.legend(
    handles=legend_handles,
    title='Baseline method',
    title_fontsize='20',
    loc='upper center',
    ncol=len(contrast_methods),
    fontsize=20,
    bbox_to_anchor=(0.5, 0.95),
    frameon=True,
    facecolor='white',
    edgecolor='black',
)

plt.suptitle(
    'Agreement and Overlap between ICUSDP and Baseline Models',
    fontsize=22,
    fontweight='bold',
    y=0.98,
)

plt.tight_layout(rect=[0, 0.01, 1, 0.92])

ridge_fig_path = os.path.join(output_dir, 'Project_Level_Ridge_Plot.png')
ridge_pdf_path = os.path.join(output_dir, 'Project_Level_Ridge_Plot.pdf')
# plt.savefig(ridge_fig_path, dpi=300, bbox_inches='tight')
plt.savefig(ridge_fig_path, dpi=600, bbox_inches='tight')
plt.savefig(ridge_pdf_path, format='pdf', bbox_inches='tight')
plt.close()

print(f'✅ now :\n   - PNG: {ridge_fig_path}\n   - PDF: {ridge_pdf_path}')