import numpy as np


def CLAMI(test_data):
    """
    CLAMI算法实现
    返回: [trainX, trainY, testX_feature]
    按照TCL格式修改
    """
    if test_data.size == 0 or len(test_data.shape) < 2:
        return [np.array([]), np.array([]), np.array([])]

    n, dim = test_data.shape

    # 确保数据是数值类型
    test_data = test_data.astype(float)

    # 计算阈值
    threshold = np.median(test_data, axis=0)

    # 聚类：为每个样本创建二进制编码
    index = np.zeros((n, dim))
    for i in range(n):
        idx = test_data[i, :] > threshold
        index[i, idx] = 1

    # 计算每个样本的"1"的数量
    count = np.sum(index, axis=1)
    unicount = np.unique(count)
    num = len(unicount)

    # 分组
    clusters = []
    for uc in unicount:
        clusters.append(np.where(count == uc)[0])

    # 标签簇
    k = np.ceil(num / 2).astype(int)

    # 处理空簇的情况
    if k > 0 and k <= len(clusters):
        # 缺陷簇（后半部分）
        if k < len(clusters):
            defective_idx = []
            for i in range(k, len(clusters)):
                defective_idx.extend(clusters[i])
            defective = np.array(defective_idx)
        else:
            defective = np.array([])

        # 非缺陷簇（前半部分）
        nondefective_idx = []
        for i in range(min(k, len(clusters))):
            nondefective_idx.extend(clusters[i])
        nondefective = np.array(nondefective_idx)
    else:
        defective = np.array([])
        nondefective = np.array([])

    # 准备训练数据
    if len(defective) > 0:
        defX = test_data[defective, :]
    else:
        defX = np.array([]).reshape(0, dim)

    if len(nondefective) > 0:
        nondefX = test_data[nondefective, :]
    else:
        nondefX = np.array([]).reshape(0, dim)

    # 合并数据
    if len(defective) > 0 and len(nondefective) > 0:
        newX = np.vstack([defX, nondefX])
        newLabel = np.hstack([np.ones(len(defective)), np.zeros(len(nondefective))])
    elif len(defective) > 0:
        newX = defX
        newLabel = np.ones(len(defective))
    elif len(nondefective) > 0:
        newX = nondefX
        newLabel = np.zeros(len(nondefective))
    else:
        return [np.array([]), np.array([]), test_data]

    # 特征选择：计算违规分数
    if len(newX) == 0:
        return [np.array([]), np.array([]), test_data]

    mvs = np.zeros((len(newX), dim))
    for j in range(dim):
        for i in range(len(newX)):
            if newX[i, j] <= threshold[j] and newLabel[i] == 1:
                mvs[i, j] = 1
            elif newX[i, j] > threshold[j] and newLabel[i] == 0:
                mvs[i, j] = 1

    # 选择违规最少的特征
    mvss = np.sum(mvs, axis=0)

    if len(mvss) == 0:
        # 如果没有特征，返回第一个特征
        trainX = newX[:, 0:1] if newX.shape[1] > 0 else newX
        trainY = newLabel
        testX_feature = test_data[:, 0:1] if test_data.shape[1] > 0 else test_data
        return [trainX, trainY, testX_feature]

    # 找到最小违规的特征
    min_mvs = np.min(mvss)
    selected_features = np.where(mvss == min_mvs)[0]

    if len(selected_features) == 0:
        selected_features = [0]

    # 选择第一个最小违规特征
    selected_feature = selected_features[0]

    # 实例选择
    subMvs = mvs[:, selected_feature:selected_feature + 1]
    IND = np.sum(subMvs, axis=1)
    valid_indices = IND == 0

    if np.sum(valid_indices) == 0:
        # 如果没有有效实例，使用所有实例
        trainX = newX[:, selected_feature:selected_feature + 1]
        trainY = newLabel
    else:
        trainX = newX[valid_indices, selected_feature:selected_feature + 1]
        trainY = newLabel[valid_indices]

    # 测试数据特征（所有原始数据在选定特征上）
    testX_feature = test_data[:, selected_feature:selected_feature + 1]

    return [trainX, trainY, testX_feature]