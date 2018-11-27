import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
print(X_test.shape)
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
print(X_outliers.shape)

# fit the model
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)  # 对样本的预测结果为1则说明为正常值，为-1表示为异常值

train_index = []
for i, j in enumerate(y_pred_train):
    if j == 1:
        train_index.append(i)  # 获取所有正常值的索引

test_index = []
y_pred_test = clf.predict(X_test)
for i, j in enumerate(y_pred_test):
    if j == 1:
        test_index.append(i)

y_pred_outliers = clf.predict(X_outliers)
outliers_index = []
for i, j in enumerate(y_pred_outliers):
    if j == 1:
        outliers_index.append(i)

new_x_train = X_train[train_index]  # 将所有预测为正常样本重新组成新的样本集
new_x_test = X_test[test_index]
new_x_outliers = X_outliers[outliers_index]

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

# 画出各个样本集的正常值分布情况
b1 = plt.scatter(new_x_train[:, 0], new_x_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(new_x_test[:, 0], new_x_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(new_x_outliers[:, 0], new_x_outliers[:, 1], c='black',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()
