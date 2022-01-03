from matplotlib import pyplot as plt

from RF import RandomForestClassifier
import pandas as pd
import os

clf = RandomForestClassifier(n_estimator=100,
                             max_depth=7,
                             min_samples_split=5,
                             split_threshold=0.1,
                             max_features='sqrt',
                             bootstrap=True,
                             max_samples=0.7,
                             random_state=42
                             )

os.chdir("C:\\Users\\vincen\\Desktop\\material\\Machine Learning\\data")
# 注意index从1开始
loan = pd.read_csv("贷款申请样本数据表.csv", header=0, sep=',', encoding='gbk')
X = loan.iloc[:, :-1]
y = loan.iloc[:, -1]

clf.fit(X, y)

y_pred = clf.predict(X)

score = clf.score(X, y)
print(score)