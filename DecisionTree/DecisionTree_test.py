"""
@author:vincen
@email:vincen.nwu@gmail.com
@Northwest University——China
@time:9:48 pm Thursday, 24 September 2020
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


#%%
import pandas as pd
import os
from DecisionTree import DecisionClassifier

def createTrainSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    dataSet = pd.DataFrame([[0, 0, 0, 0, 'N'],
                [0, 0, 0, 1, 'N'],
                [1, 0, 0, 0, 'Y'],
                [2, 1, 0, 0, 'Y'],
                [2, 2, 1, 0, 'Y'],
                [2, 2, 1, 1, 'N'],
                [1, 2, 1, 1, 'Y']],columns = ['outlook', 'temperature', 'humidity', 'windy','label'])
    return dataSet

def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = pd.DataFrame([[0, 1, 0, 0],
                [0, 2, 1, 0],
                [2, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [2, 1, 0, 1]],columns = ['outlook', 'temperature', 'humidity', 'windy'])
    return testSet

dataset = createTrainSet()
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
clf = DecisionClassifier(max_depth = 4, criterion = 'c4.5')
clf.fit(X, y, show_graph = True)
print(clf.tree)


# %%
# 最大深度的测试
import pandas as pd
import os
from DecisionTree import DecisionClassifier

os.chdir("C:\\Users\\vincen\\Desktop\\material\\Machine Learning\\data")
# 注意index从1开始
xigua = pd.read_csv("xigua.csv", header = 0, sep = ',', encoding = 'gbk')

X = xigua.iloc[:, :-1]
y = xigua.iloc[:, -1]
clf = DecisionClassifier(max_depth = 7, criterion = 'c4.5')
clf.fit(X, y, show_graph = True)
print(clf.tree)

clf = DecisionClassifier(max_depth = 4, criterion = 'c4.5')
clf.fit(X, y, show_graph = True)
print(clf.tree)
y_predict = clf.predict(X)
clf.score(X, y)


# %%
