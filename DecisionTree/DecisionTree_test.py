"""
@author:vincen
@email:vincen.nwu@gmail.com
@Northwest University——China
@time:6:48 PM Tuesday, September 15, 2020
@Compiler Environment:vscode
@Test data source:https://blog.csdn.net/u012421852/article/details/79808307
@my github address: https://github.com/vincen-github/Machine-Learning-Code
"""
# %%
import pandas as pd
import os
from DecisionTree import DecisionClassifier

os.chdir("C:\\Users\\vincen\\Desktop\\material\\Machine Learning\\data")
# 注意index从1开始
loan = pd.read_csv("贷款申请样本数据表.csv", header = 0, sep = ',', encoding = 'gbk')
X = loan.iloc[:, :-1]
y = loan.iloc[:, -1]
Xtrain = X.sample(frac = 0.8, random_state = 42, axis = 0, replace = False)
Xtest = X.drop(Xtrain.index, inplace = False)
ytrain = y.loc[Xtrain.index]
ytest = y.drop(Xtrain.index, inplace = False)

clf = DecisionClassifier(max_depth = 7, criterion = "id3")
clf.fit(X, y, show_graph = True)



# %%
