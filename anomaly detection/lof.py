import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)

# Generate train data
X = 0.3 * np.random.randn(100, 2)
# Generate some abnormal novel observations
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X + 2, X - 2, X_outliers]  # 行连接 shape(220,2)

# fit the model
clf = LocalOutlierFactor(n_neighbors=20)
y_pred = clf.fit_predict(X)  # 预测为1则为正常样本，-1为异常样本
outlier = []
for i, j in enumerate(y_pred):
    if j == 1:
        outlier.append(i)  # 获取所有正常样本

y_pred_outliers = y_pred[200:]

# plot the level sets of the decision function
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])  # 画出决策边界
Z = Z.reshape(xx.shape)

# 画出正常样本和异常样本分布
plt.subplot(211)
plt.title("Local Outlier Factor (LOF)")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)  # 决策出不同区域用不同颜色

a = plt.scatter(X[:200, 0], X[:200, 1], c='white',
                edgecolor='k', s=20)
b = plt.scatter(X[200:, 0], X[200:, 1], c='red',
                edgecolor='k', s=20)
plt.legend([a, b],
           ["normal observations",
            "abnormal observations"],
           loc="upper left")

# 画出去除LOF预测为异常样本后剩下的样本分布
plt.subplot(212)
plt.title("remove noise samples")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)  # 决策出不同区域用不同颜色
plt.scatter(X[:200, 0], X[:200, 1], c='white', edgecolor='k', s=20)

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))

plt.show()
