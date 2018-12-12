from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

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

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)

dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)

# 此处我们使用决策树作为基函数，由于存在十种标签，所以决策树的叶子数和最大深度至大要调到10
# 随着迭代增多精确率会升高，但是1000次的准确率仅为0.98，和50次相比准确率仅提高了0.1，得不偿失。
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10),
                         n_estimators=50)
ada.fit(X_train, y_train)


def echo_report(models):
    for model in models:
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        print(str(model.__class__) + "正确率为: {}".format(score), '\n',
              classification_report(y_true=y_test, y_pred=y_pred))


target_model = [rf, dt, ada]
echo_report(target_model)
