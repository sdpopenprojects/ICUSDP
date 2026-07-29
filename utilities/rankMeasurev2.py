import numpy as np


# effort-aware performance measures
def rank_measure(predict_score, effort, test_label, opt=0):
    length = len(test_label)
    if 0 in effort:
        # for avoiding effort has zero
        effort = effort + 1

    # 【修正】使用 .astype(int) 确保不污染原始连续得分 predict_score，保证后续密度排序准确
    predict_label = (predict_score >= 0.5).astype(int)

    # predict defect density
    pred_density = predict_score / effort
    actual_density = test_label / effort

    if opt == 1:  # ManualDown, ManualUp methods
        data = np.zeros(shape=(len(test_label), 5))
        data[:, 0] = predict_label
        data[:, 1] = pred_density
        data[:, 2] = test_label
        data[:, 3] = actual_density
        data[:, 4] = effort

        # actual model
        data_mdl = sorted(data, key=lambda x: (-x[0]))  # x[0]:predict_label
        data_mdl = np.array(data_mdl)
        mdl = computeArea(data_mdl, length)
    else:
        # combining
        data = np.zeros(shape=(len(test_label), 5))
        data[:, 0] = predict_label
        data[:, 1] = pred_density
        data[:, 2] = test_label
        data[:, 3] = actual_density
        data[:, 4] = effort

        # actual model(CBS+)
        data_mdl = sorted(data, key=lambda x: (-x[0], -x[1]))
        data_mdl = np.array(data_mdl)
        mdl = computeArea(data_mdl, length)

    # optimal model
    data_opt = sorted(data, key=lambda x: (-x[3], x[4]))
    data_opt = np.array(data_opt)
    opt = computeArea(data_opt, length)

    # worst model
    data_wst = sorted(data, key=lambda x: (x[3], -x[4]))
    data_wst = np.array(data_wst)
    wst = computeArea(data_wst, length)

    if opt - wst != 0:
        Popt = 1 - (opt - mdl) / (opt - wst)
    else:
        Popt = 0.5

    # 核心评估函数：返回 c系列(代码行20%) 和 m系列(模块数20%) 的基础指标
    (cErecall, cEprecision, cEfmeasure, cMCC, cPMI, cIFA, cPCI, c_ROI_PII, c_ROI_PCI, ceIFA,
     mRecall, mPrecision, mfmeasure, mMCC, mPMI, mIFA, mPCI, m_ROI_PII, m_ROI_PCI, meIFA) = computeMeasure(data_mdl, length)

    # --- 1. 基于 20% 代码行 (c系列) 的 ROI ---
    # 衡量：在 20% Effort 预算下的投资回报率
    # c_ROI_PII = cErecall / cPMI if cPMI != 0 else 0
    # c_ROI_PCI = cErecall / cPCI if cPCI != 0 else 0

    # --- 2. 基于 20% 模块数 (m系列) 的 ROI ---
    # 衡量：在检查 20% 数量文件时的投资回报率
    # m_ROI_PII = mRecall / mPMI if mPMI != 0 else 0
    # m_ROI_PCI = mRecall / mPCI if mPCI != 0 else 0

    # 返回所有指标（共 21 个返回值）
    return (Popt, cErecall, cEprecision, cEfmeasure, cMCC, cPMI, cIFA, cPCI, c_ROI_PII, c_ROI_PCI, ceIFA,
            mRecall, mPrecision, mfmeasure, mMCC, mPMI, mIFA, mPCI, m_ROI_PII, m_ROI_PCI, meIFA)


def computeMeasure(data, length):
    cumXs = np.cumsum(data[:, 4])  # 累计代码行
    cumYs = np.cumsum(data[:, 2])  # 累计缺陷数
    total_effort = cumXs[length - 1]
    total_bugs = cumYs[length - 1]
    Xs = cumXs / total_effort  # 代码行占比

    # 1. 基于 20% 代码行 (Effort-based, 标记为 c)
    # 找到第一个包含缺陷的模块下标 (Python 下标从 0 开始)
    # 对应公式中的 y，即检查到第 y 个模块时发现了首个缺陷
    idx_c = np.min(np.where(Xs >= 0.2))
    pos_c = idx_c + 1 # 转换为序数（从1开始）

    # cMCC 计算
    tp_c = cumYs[idx_c]
    fp_c = pos_c - tp_c
    fn_c = total_bugs - tp_c
    tn_c = (length - pos_c) - fn_c
    num_c = (tp_c * tn_c) - (fp_c * fn_c)
    den_c = np.sqrt((tp_c + fp_c) * (tp_c + fn_c) * (tn_c + fp_c) * (tn_c + fn_c))
    cMCC = num_c / den_c if den_c != 0 else 0

    cErecall = cumYs[idx_c] / total_bugs if total_bugs != 0 else 0
    cEprecision = cumYs[idx_c] / pos_c
    cEfmeasure = (2 * cErecall * cEprecision / (cErecall + cEprecision)) if (cErecall + cEprecision) != 0 else 0
    cPMI = pos_c / length
    # 这里的 cPCI 即为截断点实际的 Xs 占比
    cPCI = cumXs[idx_c] / total_effort

    # 衡量：在 20% Effort 预算下的投资回报率
    c_ROI_PMI = tp_c / cPMI if cPMI != 0 else 0
    c_ROI_PCI = tp_c / cPCI if cPCI != 0 else 0

    # 2. 基于 20% 模块数 (Module-based, 标记为 m)
    # 找到第一个包含缺陷的模块下标 (Python 下标从 0 开始)
    # 对应公式中的 y，即检查到第 y 个模块时发现了首个缺陷
    idx_m = int(length * 0.2) - 1
    if idx_m < 0: idx_m = 0
    pos_m = idx_m + 1 # 转换为序数（从1开始）

    # mMCC 计算
    tp_m = cumYs[idx_m]
    fp_m = pos_m - tp_m
    fn_m = total_bugs - tp_m
    tn_m = (length - pos_m) - fn_m
    num_m = (tp_m * tn_m) - (fp_m * fn_m)
    den_m = np.sqrt((tp_m + fp_m) * (tp_m + fn_m) * (tn_m + fp_m) * (tn_m + fn_m))
    mMCC = num_m / den_m if den_m != 0 else 0

    mRecall = cumYs[idx_m] / total_bugs if total_bugs != 0 else 0
    mPrecision = cumYs[idx_m] / pos_m
    mfmeasure = (2 * mRecall * mPrecision / (mRecall + mPrecision)) if (mRecall + mPrecision) != 0 else 0
    mPMI = pos_m / length
    # 这里的 mPCI 为基于模块截断时对应的累计代码占比
    mPCI = cumXs[idx_m] / total_effort

    # 衡量：在检查 20% 数量模块时的投资回报率
    m_ROI_PMI = tp_m / mPMI if mPMI != 0 else 0
    m_ROI_PCI = tp_m / mPCI if mPCI != 0 else 0

    # ==========================================
    # 3. 初始故障定位指标 (IFA / PCI / eIFA)
    # ==========================================
    if np.all(cumYs == 0):
        cIFA, ceIFA = -1, 0.0
        mIFA, meIFA = -1, 0.0
    else:
        Iidx = np.min(np.where(cumYs >= 1))
        PII_IFA = (Iidx + 1) / length # 计算 PII_IFA (公式 6): 检查模块数 / 总模块数
        PCI_IFA = cumXs[Iidx] / total_effort # 发现首个缺陷时所花费的代码行占比 (PCI),累计检查行数 / 总行数

        # 4. 计算 eIFA (公式 7): 默认 alpha = 0.5
        alpha = 0.5
        common_eIFA = alpha * PII_IFA + (1 - alpha) * PCI_IFA
        # common_eIFA = 0.5 * PII_IFA + 0.5 * PCI_IFA

        cIFA, ceIFA = float(Iidx), common_eIFA
        mIFA, meIFA = float(Iidx), common_eIFA

    return (cErecall, cEprecision, cEfmeasure, cMCC, cPMI, cIFA, cPCI, c_ROI_PMI, c_ROI_PCI, ceIFA,
            mRecall, mPrecision, mfmeasure, mMCC, mPMI, mIFA, mPCI, m_ROI_PMI, m_ROI_PCI, meIFA)


def computeArea(data, length):
    data = np.array(data)
    cumXs = np.cumsum(data[:, 4])
    cumYs = np.cumsum(data[:, 2])
    Xs, Ys = cumXs / cumXs[length - 1], cumYs / cumYs[length - 1]

    fix_subareas = [0] * len(Ys)
    fix_subareas[0] = 0.5 * Ys[0] * Xs[0]
    for i in range(1, len(Ys)):
        fix_subareas[i] = 0.5 * (Ys[i - 1] + Ys[i]) * abs(Xs[i] - Xs[i - 1])
    return sum(fix_subareas)