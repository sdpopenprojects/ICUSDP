import os
import numpy as np
import pandas as pd


folder_path = r"E:\ICUSDP\INTC\ICUSDP-main\new\FS\disscussion\FS compare\ICUSDP_FS"


def calculate_cross_project_overlap(csv_path):

  df = pd.read_csv(csv_path, header=None)
  project_sets = []


  for idx, row in df.iterrows():
    row_features = set()
    for val in row:
      if not pd.isna(val):
        for part in str(val).split(','):
          feat = part.strip()
          if feat:
            row_features.add(feat)
    project_sets.append(row_features)


  n = len(project_sets)
  pairwise_scores = []
  for i in range(n):
    for j in range(i + 1, n):
      s_i = project_sets[i]
      s_j = project_sets[j]

      intersection = len(s_i.intersection(s_j))
      union = len(s_i.union(s_j))


      sim = intersection / union if union > 0 else 1.0
      pairwise_scores.append(sim)

  mean_overlap = np.mean(pairwise_scores)
  std_overlap = np.std(pairwise_scores)
  return mean_overlap, std_overlap, n



files_to_process = {
    "With_FS (Top-1)": os.path.join(folder_path, "top1_v1.csv"),
    "With_FS (Top-3)": os.path.join(folder_path, "top3_v1.csv"),

}

results = []
for label, filepath in files_to_process.items():
  if os.path.exists(filepath):
    mean_val, std_val, count = calculate_cross_project_overlap(filepath)
    results.append({
        "Setting": label,
        "Mean_Overlap": round(mean_val, 4),
        "Std_Overlap": round(std_val, 4),
        "Total_Projects": count,
    })
    print(
        f"[{label}] : Mean Overlap = {mean_val:.4f} (± {std_val:.4f})"
    )
  else:
    print(f"error: {filepath}")


if results:
  result_df = pd.DataFrame(results)
  out_path = os.path.join(folder_path, "overlap_summary_results.csv")
  result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
  print(f"\n: {out_path}")