import numpy as np
import pandas as pd

def ONE(LOC, true_label, exclude_pct=20, cutoff_pct=0.2):
    df = pd.DataFrame({
        'original_idx': np.arange(len(LOC)),
        'loc': LOC,
        'label': true_label
    })

    # 1. 降序排列
    # descending by LOC first, then ascending by actual label if LOCs are equal
    df_sorted = df.sort_values(by=['loc', 'label'], ascending=[False, True]).reset_index(drop=True)

    total_loc = df_sorted['loc'].sum()
    df_sorted['cumsum_loc'] = df_sorted['loc'].cumsum()

    # 2. 找到大模块并排除移动到末尾
    # finding the larger modules (e.g., top 20%)
    exclude = total_loc / 100 * exclude_pct
    valid_excludes = df_sorted[df_sorted['cumsum_loc'] <= exclude]

    if valid_excludes.empty:
        sub1 = pd.DataFrame(columns=df_sorted.columns)
        remain_data = df_sorted.copy()
    else:
        idx = valid_excludes.index.max()
        sub1 = df_sorted.iloc[:idx + 1].copy()
        remain_data = df_sorted.iloc[idx + 1:].copy()

        # for larger modules: ascending by LOC first, then ascending by actual label if LOCs are equal
        sub1 = sub1.sort_values(by=['loc', 'label'], ascending=[True, True])

    # 拼接
    # catenate
    final_data = pd.concat([remain_data, sub1], ignore_index=True)

    # 3. 打预测标签
    # according to cutoff_pct to set predict label
    final_data['predict_label'] = 0
    final_data['new_cumsum_loc'] = final_data['loc'].cumsum()

    if cutoff_pct > 1:  # **% LOC# 按代码量百分比阈值
        cutoff = total_loc / 100 * cutoff_pct
        valid_cutoffs = final_data[final_data['new_cumsum_loc'] <= cutoff]
        if not valid_cutoffs.empty:
            cutoff_idx = valid_cutoffs.index.max()
            final_data.loc[:cutoff_idx, 'predict_label'] = 1
    else:  # **% modules# 按模块个数百分比阈值
        cutoff_idx = np.floor(len(LOC) * cutoff_pct)
        final_data.loc[:cutoff_idx - 1, 'predict_label'] = 1

    final_data.sort_values(by=['original_idx'], ascending=[True])

    # 1. 核心修复：在这里通过 original_idx 严格恢复到最原始的、和外层 test_label 行行对齐的物理序列！
    final_data_restored = final_data.sort_values(by=['original_idx'], ascending=[True]).reset_index(drop=True)

    # 2. 严格返回这个恢复原序后的 DataFrame
    return final_data_restored

    # return final_data