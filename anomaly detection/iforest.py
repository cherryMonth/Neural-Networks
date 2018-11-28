import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

import seaborn as sns

sns.set()
"""
数据集是计算机的监控数据，主要包含两个特征Latency(延迟)和Throughput(吞吐量)。
目标为是否是异常数据: 0为假，1为真。
"""

X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
Y_test = pd.read_csv('data/Y_test.csv')

# fit the model
clf = IsolationForest()
clf.fit(X_train)

normal_index = list()
abnormal_index = list()
y_pred_test = clf.predict(X_test)  # 对样本的预测结果为1则说明为正常值，为-1表示为异常值
for i, j in enumerate(y_pred_test):
    if j == 1:
        normal_index.append(i)  # 存储正常样本的索引
    else:
        abnormal_index.append(i)  # 存储异常样本的索引

normal_series = X_test.loc[normal_index]  # 获取对应样本集合
abnormal_series = X_test.loc[abnormal_index]

real_normal_index = list()
real_abnormal_index = list()
for i, j in enumerate(np.array(Y_test)):
    if j == 0:  # 真实样本0为正常，1为异常
        real_normal_index.append(i)  # 存储真实正常样本的索引
    else:
        real_abnormal_index.append(i)  # 存储真实异常样本的索引

real_normal_series = X_test.loc[real_normal_index]  # 获取对应样本集合
real_abnormal_series = X_test.loc[real_abnormal_index]

# 向量是根据各个特征的最大最小值然后平均划分点得到
# 然后把各个向量之间做叉乘，变成一个二维平面，用于绘制等高线
x_series = np.linspace(np.min(X_test['Latency']) - 2, np.max(X_test['Latency']) + 2, 50)
y_series = np.linspace(np.min(X_test['Throughput']) - 2, np.max(X_test['Throughput']) + 2, 50)
xx, yy = np.meshgrid(x_series, y_series)
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 接下来画等高线
plt.subplot(211)
plt.title("predict IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(normal_series['Latency'], normal_series['Throughput'], c='white', s=20, edgecolor='k')
c = plt.scatter(abnormal_series['Latency'], abnormal_series['Throughput'], c='red', s=20, edgecolor='k')

plt.subplot(212)
plt.title("real IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b2 = plt.scatter(real_normal_series['Latency'], real_normal_series['Throughput'], c='white', s=20, edgecolor='k')
c2 = plt.scatter(real_abnormal_series['Latency'], real_abnormal_series['Throughput'], c='red', s=20, edgecolor='k')
plt.tight_layout(h_pad=1)
plt.savefig('iforest.png')
plt.show()
