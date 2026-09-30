import numpy as np
from numpy import int64


# calculate instance violation ratio
# data: data set
# MVR: p number of selected metrics with minimum MVR
# DL: data label by using TCL method
# DT: selected threshold corresponding selected metrics
def IVR(data, MVR, DL, DT):
    # number of samples * number of features
    [N, M] = data.shape

    # selected p metrics with minimum MVR
    p = np.ceil(np.log2(M))
    p = p.astype(int64)

    index = np.argsort(MVR)
    sidx = index[0:p]
    sort_data = data.iloc[:, sidx]
    sort_DT = DT[sidx]

    # calculate IVR of each metric
    iVR = np.zeros(N)
    for i in range(N):
        # number of violated instance values for defective
        cy = 0
        # number of violated instance values for non-defective
        cn = 0

        for j in range(p):
            if DL[i] == 1:
                if sort_data.iloc[i, j] <= sort_DT[j]:
                    r = 1
                else:
                    r = 0
                cy = cy + r

            if DL[i] == 0:
                if sort_data.iloc[i, j] > sort_DT[j]:
                    s = 1
                else:
                    s = 0
                cn = cn + s

        iVR[i] = (cy + cn) / p

    return iVR
