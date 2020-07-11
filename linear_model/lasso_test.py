from Multiple_linear_regression import Lasso
import numpy as np
import pandas as pd

shape = [200,8]
mean = [1,2,3,4,5,6,7,8]
cov = np.diag(np.ones((8)))

# beta[-1] 为 bias
beta = [1,0,3,0,2,0,2,0,0]
# print(mean,'\n',cov)
# 实例化
lasso = Lasso(c = 0.3,shape = shape,mean = mean,cov = cov,noise_miu = 0,noise_sigma = 1,C = 1)
# 生成数据集
dataset = lasso.generate_data(beta,random_state = 42)

print(dataset.head())

alpha = lasso.iterate_ridge(dataset,max_epoch = 500,tol = 5e-2)

X = np.asarray(dataset.iloc[:,:-1])
y = np.asarray(dataset.iloc[:,-1])

print("the estimate of lasso:",alpha)
print("the error of lasso:",lasso.score(dataset))


from Multiple_linear_regression import LinearRegression

lr = LinearRegression()
beta,_ = lr.fit(dataset)
print("the estimate of OLS:",beta.reshape(-1,))
print("the error of linear regression:",((y - np.dot(X,list(beta)).reshape(y.shape))**2).sum())



# lasso = Lasso(c = 0.3,shape = shape,mean = mean,cov = cov,noise_miu = 0,noise_sigma = 1,C = 1)
# # from sklearn.preprocessing import StandardScaler
# # standardscaler = StandardScaler()
# # X = pd.DataFrame(standardscaler.fit_transform(X))
# # dataset = pd.concat([X,pd.Series(y)],axis = 1)

# alpha = lasso.coordinate_descent(dataset,eps = 1)

# print(alpha)
# print(lasso.score(dataset))


