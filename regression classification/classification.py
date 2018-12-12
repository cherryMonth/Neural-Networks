import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

"""
在线广告，预测用户是否购买
"""

dataset = pd.read_csv('data/network_ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

svc = SVC(random_state=0)
svc.fit(X_train, y_train)

log = LogisticRegression(random_state=0)
log.fit(X_train, y_train)

bayes = GaussianNB()
bayes.fit(X_train, y_train)


def echo_report(models):
    for model in models:
        score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        print(str(model.__class__) + "正确率为: {}".format(score), '\n',
              classification_report(y_true=y_test, y_pred=y_pred))


target_model = [svc, log, bayes]
echo_report(target_model)
