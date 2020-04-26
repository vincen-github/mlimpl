import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Multiple_linear_regression import LinearRegression
from Multiple_linear_regression import RidgeRegression
from Multiple_linear_regression import Lasso


# README
# help(LinearRegression)

shape = [100,5]
mean = [1,2,3,4,5]
#正态分布不相关与独立等价，设置cov = I 相当于说明生成的正态分布随机变量彼此独立,理论上来说，这样生成的特征间彼此间共线性很弱
cov = np.diag(np.ones(5))
# print(cov.shape[0])
# beta[-1] 为 bias
beta = [1,2,3,4,5,1]
# print(mean,'\n',cov)
# 实例化
lr = LinearRegression(c = 0.1,shape = shape,mean = mean,cov = cov,noise_miu = 0,noise_sigma = 1)
# 生成数据集
dataset = lr.generate_data(beta)
# print(dataset)
# 计算线性回归的条件数，值为60左右,说明特征间不存在明显的共线性
print(lr.regression_cond())
# 生成相关系数矩阵反映特征间的共线情况，得到与上相同的结论
print(lr.regression_correlation())
# 训练
alpha,training_error = lr.fit(dataset)
# 打印估计出的参数和训练误差
print('LinearRegression Estimated parameter value：',alpha)
print('training_error:',training_error)

# 利用sklearn划分训练集和测试集测试score
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(dataset.iloc[:,:-1],
                                                dataset.iloc[:,-1],
                                                test_size = 0.2,
                                                random_state = 42
                                                )
training_data = pd.DataFrame(np.hstack([Xtrain,np.asarray(ytrain).reshape(-1,1)]))
testing_data = pd.DataFrame(np.hstack([Xtest,np.asarray(ytest).reshape(-1,1)]))

alpha,training_error = lr.fit(training_data)
print('LinearRegression Estimated parameter value：',alpha)
print('LinearRegression training_error:',training_error)
_,testing_error = lr.score(testing_data)
print('LinearRegression testing_error:',testing_error)

y_predict = lr.predict(Xtest)
print(y_predict)


#RidgeRegression test
# 实例化对象
ridge = RidgeRegression(c = 0.1,shape = shape,mean = mean,cov = cov,noise_miu = 0,noise_sigma = 1)
# 生成数据
dataset = ridge.generate_data(beta)
#划分训练集测试集
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(dataset.iloc[:,:-1],
                                                dataset.iloc[:,-1],
                                                test_size = 0.2,
                                                random_state = 42
                                                )
training_data = pd.DataFrame(np.hstack([Xtrain,np.asarray(ytrain).reshape(-1,1)]))
testing_data = pd.DataFrame(np.hstack([Xtest,np.asarray(ytest).reshape(-1,1)]))

#test
alpha,training_error = ridge.fit(training_data)
print('RidgeRegression Estimated parameter value：',alpha)
print('RidgeRegression training_error:',training_error)
testing_error,_ = ridge.score(testing_data)
print('RidgeRegression testing_error:',testing_error)

y_predict = ridge.predict(Xtest)
print(y_predict)

# 结论，数据非病态下,RidgeRegression与LinearRegression表现差异不大

# 线性回归批量梯度下降测试
# 实例化
lr = LinearRegression(c = 0.1,shape = shape,mean = mean,cov = cov,noise_miu = 0,noise_sigma = 1)
# 生成数据集
dataset = lr.generate_data(beta)
alpha,training_error_list = lr.batch_gradient_descent(max_epoch = 70,eta = 0.00001,show_result = True)
print('LinearRegression Estimated parameter value：',alpha)
print('LinearRegression min training_error_list:',min(training_error_list))

#Ridge批量梯度下降测试
ridge = RidgeRegression(c = 0.1,shape = shape,mean = mean,cov = cov,noise_miu = 0,noise_sigma = 1,C = 1)
# 生成数据集
dataset = ridge.generate_data(beta)
alpha,training_error_list = ridge.batch_gradient_descent(max_epoch = 70,eta = 0.00001,show_result = True)
print('RidgeRegression Estimated parameter value：',alpha)
print('RidgeRegression min training_error_list:',min(training_error_list))


# AdamOptimizer
# 非凸简单问题AdamOptimizer的下降速度比batch gradient descent慢
shape = [100,5]
mean = [1,2,3,4,5]
cov = np.diag(np.ones(5))
beta = [1,2,3,4,5,1]
lr = LinearRegression(c = 0.1,shape = shape,mean = mean,cov = cov,noise_miu = 0,noise_sigma = 1)
dataset = lr.generate_data(beta)
alpha,training_error_list = lr.AdamOptimizer(eta = 0.1,max_epoch = 1000,show_result = True)
print(alpha)
print(min(lr.error_list))
# 查看valley处的loss变化情况
import seaborn as sns
sns.set()
plt.figure(dpi = 110)
plt.plot([x for x in range(200,998)],training_error_list[200:-1],c = 'b',label = '$loss \quad curve$')
plt.title('$Adam\quad Optimizer \quad error$')
plt.xlabel('$epoch$')
plt.ylabel('$error$')
plt.legend()
plt.show()


#Ridge批量梯度下降测试
ridge = RidgeRegression(c = 0.1,shape = shape,mean = mean,cov = cov,noise_miu = 0,noise_sigma = 1,C = 1)
# 生成数据集
dataset = ridge.generate_data(beta)
alpha,training_error_list = ridge.AdamOptimizer(max_epoch = 1000,eta = 0.1,show_result = True)
print('Adaptive moment estimated(RidgeRegression) parameter value：',alpha)
print('Adaptive moment estimated(RidgeRegression) min training_error_list:',min(training_error_list))
# 查看valley处的loss变化情况
import seaborn as sns
sns.set()
plt.figure(dpi = 110)
plt.plot([x for x in range(200,998)],training_error_list[200:-1],c = 'b',label = '$loss \quad curve$')
plt.title('$Adam\quad Optimizer \quad error$')
plt.xlabel('$epoch$')
plt.ylabel('$error$')
plt.legend()
plt.show()