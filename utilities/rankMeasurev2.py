import numpy as np


# effort-aware performance measures
def rank_measure(predict_score, effort, test_label, opt=0):
    length = len(test_label)
    if 0 in effort:
        # for avoiding effort has zero
        effort = effort + 1


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


    (cErecall, cEprecision, cEfmeasure, cMCC, cPMI, cIFA, cPCI, c_ROI_PII, c_ROI_PCI, ceIFA,
     mRecall, mPrecision, mfmeasure, mMCC, mPMI, mIFA, mPCI, m_ROI_PII, m_ROI_PCI, meIFA) = computeMeasure(data_mdl, length)




    return (Popt, cErecall, cEprecision, cEfmeasure, cMCC, cPMI, cIFA, cPCI, c_ROI_PII, c_ROI_PCI, ceIFA,
            mRecall, mPrecision, mfmeasure, mMCC, mPMI, mIFA, mPCI, m_ROI_PII, m_ROI_PCI, meIFA)


def computeMeasure(data, length):
    cumXs = np.cumsum(data[:, 4])
    cumYs = np.cumsum(data[:, 2])
    total_effort = cumXs[length - 1]
    total_bugs = cumYs[length - 1]
    Xs = cumXs / total_effort


    idx_c = np.min(np.where(Xs >= 0.2))
    pos_c = idx_c + 1


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

    cPCI = cumXs[idx_c] / total_effort


    c_ROI_PMI = tp_c / cPMI if cPMI != 0 else 0
    c_ROI_PCI = tp_c / cPCI if cPCI != 0 else 0


    idx_m = int(length * 0.2) - 1
    if idx_m < 0: idx_m = 0
    pos_m = idx_m + 1


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

    mPCI = cumXs[idx_m] / total_effort


    m_ROI_PMI = tp_m / mPMI if mPMI != 0 else 0
    m_ROI_PCI = tp_m / mPCI if mPCI != 0 else 0


    if np.all(cumYs == 0):
        cIFA, ceIFA = -1, 0.0
        mIFA, meIFA = -1, 0.0
    else:
        Iidx = np.min(np.where(cumYs >= 1))
        PII_IFA = (Iidx + 1) / length
        PCI_IFA = cumXs[Iidx] / total_effort


        alpha = 0.5
        common_eIFA = alpha * PII_IFA + (1 - alpha) * PCI_IFA


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