from sklearn.model_selection import train_test_split
import pandas as pd

"""
sklearn 自带的手写数字数据集
数字是以如下形式的二维矩阵，标签即为矩阵所代表的数字

[010]
[010]  - > 数字1
[010]

[111]
[010]  - > 数字2
[111]

"""

from sklearn.datasets import load_digits

X = pd.DataFrame(load_digits()['data'])
y = pd.DataFrame(load_digits()['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_true=y_test, y_pred=y_pred))

