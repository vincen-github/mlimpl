import os
import pandas as pd
from naive_bayes import NaiveBayes
os.chdir("C:\\Users\\vincen\\Desktop")
xigua = pd.read_csv('xigua.csv', header = 0, encoding = 'gbk', engine = 'python')

#抽样划分训练集和测试集
xigua_test = xigua.sample(frac = 0.3, replace = False, axis = 0, random_state = 42)
xigua_train = xigua.drop(xigua_test.index, inplace = False)
Xtrain = xigua_train.iloc[:, :-1]
ytrain = xigua_train.iloc[:, -1]
Xtest = xigua_test.iloc[:, :-1]
ytest = xigua_test.iloc[:, -1]

clf = NaiveBayes()
y_predict = clf.MultinomialNB(X = Xtrain, y = ytrain, Xtest = Xtest, laplacian_smoothing = True, lambd = 1)
print(ytest)
print(y_predict)

print("the accuracy of testing data:", clf.score_by_accuracy(ytest))

y_predict = clf.MultinomialNB(X = Xtrain, y = ytrain, laplacian_smoothing = True, lambd = 1)
print(ytrain)
print(y_predict)
print("the accuracy of training data:", clf.score_by_accuracy(ytrain))



