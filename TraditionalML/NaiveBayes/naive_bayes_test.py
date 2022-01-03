import os
import pandas as pd
import numpy as np
# from naive_bayes import MultinomialNB
# os.chdir("C:\\Users\\vincen\\Desktop")
# xigua = pd.read_csv('xigua.csv', header = 0, encoding = 'gbk', engine = 'python')

# #抽样划分训练集和测试集
# xigua_test = xigua.sample(frac = 0.25, replace = False, axis = 0, random_state = 42)
# xigua_train = xigua.drop(xigua_test.index, inplace = False)
# Xtrain = xigua_train.iloc[:, :-1]
# ytrain = xigua_train.iloc[:, -1]
# Xtest = xigua_test.iloc[:, :-1]
# ytest = xigua_test.iloc[:, -1]
# clf = MultinomialNB(laplacian_smoothing = True, lambd = 1)
# clf.fit(X = Xtrain, y = ytrain)
# single_y_predict = clf.predict_single_instance(Xtest.iloc[0,:])
# y_predict = clf.predict(Xtest)
# score = clf.score(Xtest, ytest)
# print("the accuracy of testing data:", score)
# train_score = clf.score(Xtrain, ytrain)
# print("the accuracy of training data:", train_score)

# clf = MultinomialNB(laplacian_smoothing = True, lambd = 1)
# X = xigua.iloc[:,:-1]
# y = xigua.iloc[:,-1]
# clf.fit(X, y)
# score = clf.score(X, y)
# print("the accuracy of whole data:", score)

#与sklearn的训练结果比较
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.preprocessing import OneHotEncoder
# X = xigua.iloc[:,:-1]
# y = xigua.iloc[:,-1]
# onehot = OneHotEncoder()
# X = onehot.fit_transform(X)
# y = np.asarray(y).reshape(-1, 1)

# clf = MultinomialNB()
# clf.fit(X, y)
# print("*********")
# print(clf.score(X, y))


#gaussian naive bayes test
import pandas as pd
from naive_bayes import GaussianNB
gender = pd.read_csv('Gender_classification.csv', header = 0, encoding = 'utf-8')
test_sample = gender.iloc[-1,1:]
gender_droped = gender.drop(gender.shape[0] - 1, axis = 0, inplace = False)
Xtrain = gender_droped.iloc[:,1:]
ytrain = gender_droped.人

clf = GaussianNB()
# print(clf.normal_density(x = 0, mu = 0, sigma = 1))
clf.fit(Xtrain, ytrain)
# print(clf.predict_single_instance(test_sample))
print(clf.predict(Xtrain))
print("the accuracy of training data:", clf.score(Xtrain, ytrain))

# from scipy.stats import norm
# print(norm.pdf(0 , loc = 0, scale = 1))