import numpy as np


# calculate metric violation ratio
# data: data set
# DL: data label by using TCL method
# DT: metric threshold obtained from the threshold derivation method
def MVR(data, DL, DT):
    # number of samples * number of features
    [N, M] = data.shape

    # calculate MVR of each metric
    mVR = np.zeros(M)
    for i in range(M):
        # number of violated metric values for defective
        cy = 0
        # number of violated metric values for non-defective
        cn = 0

        for j in range(N):
            if DL[j] == 1:
                if data.iloc[j, i] <= DT[i]:
                    r = 1
                else:
                    r = 0
                cy = cy + r

            if DL[j] == 0:
                if data.iloc[j, i] > DT[i]:
                    s = 1
                else:
                    s = 0
                cn = cn + s

        mVR[i] = (cy + cn) / N

    return mVR
