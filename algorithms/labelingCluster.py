import numpy as np
import pandas as pd
from numpy import int64
from sklearn import preprocessing


# labeling cluster according to the metric values of modules
def labelCluster(fea, clus_label):
    fea = preprocessing.scale(fea)
    fea = pd.DataFrame(fea)
    fea['clus_label'] = clus_label
    fea1 = fea[fea['clus_label'] == 0].iloc[:, :-1]
    fea2 = fea[fea['clus_label'] == 1].iloc[:, :-1]
    mean_fea1 = fea1.mean().mean()
    mean_fea2 = fea2.mean().mean()
    if mean_fea1 > mean_fea2:
        for i, label in enumerate(clus_label):
            if clus_label[i] == 0:
                clus_label[i] = 1
            else:
                clus_label[i] = 0

    return clus_label


# labeling cluster according to the number of modules
def labelCluster_v2(clus_label):
    n1 = clus_label[clus_label == 1].size # defective
    n2 = clus_label[clus_label == 0].size # nondefective

    # cluster with the smaller number of modules is labeled defective
    if n1 > n2:
        for i, label in enumerate(clus_label):
            if clus_label[i] == 0:
                clus_label[i] = 1
            else:
                clus_label[i] = 0
    return clus_label


# labeling cluster according to the metric values of modules
def labelCluster_v3(fea, clus_label):
    fea = preprocessing.scale(fea)
    preLabel = np.zeros(len(clus_label))

    rs = np.sum(fea, axis=1)

    idx1 = clus_label > 0
    rs1 = rs[idx1]
    rc1 = np.mean(rs1)
    preLabel[idx1] = (rs1 > rc1)

    idx2 = clus_label <= 0
    rs2 = rs[idx2]
    rc2 = np.mean(rs2)
    preLabel[idx2] = (rs2 > rc2)

    return preLabel.astype(int64)