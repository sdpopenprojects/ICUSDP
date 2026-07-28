import numpy as np
import pandas as pd

def ONE(LOC, true_label, exclude_pct=20, cutoff_pct=0.2):
    df = pd.DataFrame({
        'original_idx': np.arange(len(LOC)),
        'loc': LOC,
        'label': true_label
    })

    # 1. 降序排列
    df_sorted = df.sort_values(by=['loc', 'label'], ascending=[False, True]).reset_index(drop=True)

    total_loc = df_sorted['loc'].sum()
    df_sorted['cumsum_loc'] = df_sorted['loc'].cumsum()

    # 2. 找到大模块并排除移动到末尾
    exclude = total_loc / 100 * exclude_pct
    valid_excludes = df_sorted[df_sorted['cumsum_loc'] <= exclude]

    if valid_excludes.empty:
        sub1 = pd.DataFrame(columns=df_sorted.columns)
        remain_data = df_sorted.copy()
    else:
        idx = valid_excludes.index.max()
        sub1 = df_sorted.iloc[:idx + 1].copy()
        remain_data = df_sorted.iloc[idx + 1:].copy()

        # 大模块：LOC升序，label升序
        sub1 = sub1.sort_values(by=['loc', 'label'], ascending=[True, True])

    # 拼接
    final_data = pd.concat([remain_data, sub1], ignore_index=True)

    # 3. 打预测标签
    final_data['predict_label'] = 0
    final_data['new_cumsum_loc'] = final_data['loc'].cumsum()

    # ✨【支持 0.31-0.39 灵活判断】
    if cutoff_pct > 1:  # 传入的是像 20 这样的百分比数
        cutoff = total_loc / 100 * cutoff_pct
        valid_cutoffs = final_data[final_data['new_cumsum_loc'] <= cutoff]
        if not valid_cutoffs.empty:
            cutoff_idx = valid_cutoffs.index.max()
            final_data.loc[0:cutoff_idx, 'predict_label'] = 1
        else:
            final_data.loc[0:0, 'predict_label'] = 1
    else:  # 传入的是像 0.2 或 0.31-0.39 这样的模块数占比小数
        cutoff_idx = int(len(final_data) * cutoff_pct)
        if cutoff_idx < 1: cutoff_idx = 1
        final_data.loc[0:cutoff_idx-1, 'predict_label'] = 1

    return final_data