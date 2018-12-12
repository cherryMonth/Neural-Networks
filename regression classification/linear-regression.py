from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set()

"""
年龄，性别，体质指数，血压，(六种血清的化验数据) 10个特征
预测目标为病情指数
"""

diabetes = load_diabetes()
X = diabetes['data'][:, np.newaxis, 2]
Y = diabetes['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
regr = linear_model.LinearRegression().fit(X_train, Y_train)
plt.scatter(X_test, Y_test, c='black')
plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)
plt.savefig('linear-regression.png')
plt.show()