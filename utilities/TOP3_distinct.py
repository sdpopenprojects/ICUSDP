import os
import matplotlib.pyplot as plt
import pandas as pd


importance_file = r'E:\ICUSDP\INTC\ICUSDP-main\new\FS\RQ3\feature_analysis_results_top3_groups\global_top3_selection_statistics.csv'

save_dir = r'E:\ICUSDP\INTC\ICUSDP-main\new\FS\RQ3\top3_distinct'
os.makedirs(save_dir, exist_ok=True)

save_fig = os.path.join(save_dir, 'Feature_Category_Distribution.png')
save_fig_pdf = os.path.join(save_dir, 'Feature_Category_Distribution.pdf')
save_csv = os.path.join(save_dir, 'Feature_Category_Distribution.csv')


if not os.path.exists(importance_file):
  print(f'[ERROR] : {importance_file}')
  exit()

df = pd.read_csv(importance_file)
df = df.sort_values(by='Percentage', ascending=False)

print('\nTop-3 Groups Features & Selection Percentage:')
print(df)


feature_category = {
    # 1. Code Metrics
    'PercentLackOfCohesion': 'Code Metrics',
    'CountClassCoupled': 'Code Metrics',
    'MaxNesting_Mean': 'Code Metrics',
    'RatioCommentToCode': 'Code Metrics',
    'CountLineComment': 'Code Metrics',
    'AvgCyclomaticModified': 'Code Metrics',
    'CountOutput_Mean': 'Code Metrics',
    'CountDeclInstanceVariable': 'Code Metrics',
    'CountInput_Mean': 'Code Metrics',
    'CountDeclMethodPublic': 'Code Metrics',
    'CountDeclMethodPrivate': 'Code Metrics',
    'CountDeclInstanceMethod': 'Code Metrics',
    'CountClassBase': 'Code Metrics',
    'AvgEssential': 'Code Metrics',
    'AvgLineBlank': 'Code Metrics',
    'CountDeclClassMethod': 'Code Metrics',
    # 2. Process Metrics
    'Added_lines': 'Process Metrics',
    # 3. Ownership Metrics
    'OWN_COMMIT': 'Ownership Metrics',
    'MAJOR_COMMIT': 'Ownership Metrics',
    'MAJOR_LINE': 'Ownership Metrics',
}


category_count = {
    'Code Metrics': 0,
    'Process Metrics': 0,
    'Ownership Metrics': 0,
}

unknown = []

for feature in df['Feature']:
  if feature in feature_category:
    category = feature_category[feature]
    category_count[category] += 1
  else:
    unknown.append(feature)

print('\nCategory Count (Number of Features):')
print(category_count)

if unknown:
  print('\n:')
  print(unknown)


category_df = pd.DataFrame({
    'Category': list(category_count.keys()),
    'Number': list(category_count.values()),
})

total = category_df['Number'].sum()
if total > 0:
  category_df['Percentage'] = (category_df['Number'] / total) * 100
else:
  category_df['Percentage'] = 0

category_df.to_csv(save_csv, index=False, encoding='utf-8-sig')


print(category_df)


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

plt.figure(figsize=(7, 5))

bars = plt.bar(
    category_df['Category'],
    category_df['Percentage'],
    color='steelblue',
    edgecolor='black',
    linewidth=1.2,
    width=0.45,
)

plt.title(
    'Feature Category Distribution (Top-3)',
    fontsize=18,
    fontweight='bold',
    pad=15,
)
plt.ylabel('Percentage (%)', fontsize=16, fontweight='bold')
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)


for bar, value in zip(bars, category_df['Percentage']):
  plt.text(
      bar.get_x() + bar.get_width() / 2,
      value + 1.5,
      f'{value:.1f}%',
      ha='center',
      va='bottom',
      fontsize=15,
      fontweight='bold',
  )

max_percentage = category_df['Percentage'].max()
plt.ylim(0, max_percentage + 15)

plt.tight_layout()



plt.savefig(save_fig, dpi=300, bbox_inches='tight')
plt.savefig(save_fig_pdf, format='pdf', bbox_inches='tight')
plt.close()

print('\n[SUCCESS] ')
print(f'PNG : {save_fig}')
print(f'PDF : {save_fig_pdf}')
print(f'CSV : {save_csv}')