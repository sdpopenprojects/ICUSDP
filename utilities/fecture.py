import numpy as np
import matplotlib.pyplot as plt

# 提取特征重要性信息
feature_names = X[0].columns.values
importances = rf_classifier.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制条形图
plt.bar(range(X[0].columns.shape[1]), importances[indices], align='center')

# 在每个条形图上显示特征重要性数值
for x in range(X[0].columns.shape[1]):
    text = '{:.2f}'.format(importances[indices[x]])
    plt.text(x, importances[indices[x]] + 0.01, text, ha='center')

# 设置x轴刻度标签
plt.xticks(range(X[0].columns.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X[0].columns.shape[1]])
plt.ylim(0.0, np.max(importances) + 0.05)

# 添加标签和标题
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')

# 自动调整布局并显示图形
plt.tight_layout()
plt.show()
