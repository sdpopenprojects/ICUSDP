import numpy as np
from numpy import int64
from sklearn.ensemble import RandomForestClassifier

from algorithms.InstanceVR import IVR
from algorithms.MetricVR import MVR


# Threshold Clustering Labeling Plus Method
# data: data set
# DL: data label by using TCL method
# DT: metric threshold obtained from the threshold derivation method
def TCLP(data, DL, DT):
    # number of samples * number of features
    [N, M] = data.shape

    # calculate MVR
    mvr = MVR(data, DL, DT)

    # selected p metrics with minimum MVR
    p = np.ceil(np.log2(M))
    p = p.astype(int64)
    idx_met = np.argsort(mvr)
    sel_idx_met = idx_met[0:p]
    # sel_DT = DT[sel_idx_met]

    # calculate IVR
    ivr = IVR(data, mvr, DL, DT)

    # selected q (70%) instances with minimum IVR
    ra = 0.7
    q = np.floor(N * ra)
    q = q.astype(int64)
    idx_ins = np.argsort(ivr)
    sel_idx_ins = idx_ins[0:q]

    sel_DL = DL[sel_idx_ins]
    num_def = np.sum(sel_DL == 1)
    while num_def < 2:
        ra = ra + 0.1
        q = np.floor(N * ra)
        q = q.astype(int64)
        sel_idx_ins = idx_ins[0:q]

        sel_DL = DL[sel_idx_ins]
        num_def = np.sum(sel_DL == 1)

    sel_data = data.iloc[sel_idx_ins, sel_idx_met]

    # train random forest classification model
    rf = RandomForestClassifier()
    rf.fit(sel_data, sel_DL)

    # test_data
    te_data = data.iloc[:, sel_idx_met]
    preLabel = rf.predict(te_data)

    return preLabel
