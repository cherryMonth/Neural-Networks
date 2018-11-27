import pandas as pd
from sklearn.model_selection import train_test_split

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

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print(score)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_true=y_test, y_pred=y_pred))
