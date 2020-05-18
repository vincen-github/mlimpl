# author:vincen
# date:2020/4/10
# encoding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Parent Class
class LinearRegression(object):
    '''
    LinearRegression Object(parent class):
        attribute:
            c:噪声的影响强度,取值[0,1] 默认为0
            shape:生成数据集的形状,默认为None     dtype = [sample_num,feature_num]
            mean:x所服从正态分布的均值向量       默认为None
            cov:x所服从正态分布的协方差矩阵      默认为None
            noise_miu：噪声所服从的正态分布的均值
            noise_sigma:噪声所服从的正态分布的方差
            noise默认服从标准正态分布
        function:
            generate_data(): 按照实例化对象的初始属性生成数据集
            regression_cond():计算线性回归中 X 的条件数
            regression_correlation():可视化矩阵X的相关系数矩阵
            fit():最小二乘估计训练线性模型(直接利用(X^T*X)^(-1)*X*y计算)，return alpha和RSS
            score():通过训练好的model计算测试误差
            predict():利用已训练好的model预测测试样本
            batch_gradient_descent():批量梯度下降 每一个epoch用整个训练集计算梯度
            stochastic_gradient_descent():随机梯度下降 每一个epoch中运用每一个样本计算梯度
            mini_batch_gradient_descent():小批量梯度下降 每一个epoch中运用等大小的训练集的子集计算梯度
            AdamOptimizer():Adaptive Moment Estimation
            square_R():goodness of fit
            modified_square_R():modified_square_R
    '''
    def __init__(self,c = 0,shape = None,mean = None,cov = None,noise_miu = 0,noise_sigma = 1):
            self.c = c
            self.shape = shape
            self.mean = mean
            self.cov = cov
            self.noise_miu = noise_miu
            self.noise_sigma = noise_sigma

    def generate_data(self,beta):
        """
        parameters:
            input:beta
                The parameters of data generated,the last value of beta is the bias of generative dataset.
            return : the generative dataset
                type: DataFrame
        """
        #检查beta的维数是否是特征数+1(+1是因为偏置的存在)
        if len(beta) != (self.shape)[1]+1:
            raise Exception("The dimensions of beta is not fit the shape of dataset.")
        # 检查协方差矩阵和均值向量的维数，若否，抛出异常
        elif np.asarray(self.cov).shape[0] != self.shape[1]:
            raise Exception("The dimensions of the covariance matrix and the shape of dataset are not fitting.")
        elif len(self.mean) != self.shape[1]:
            raise Exception("The dimensions of the mean vector and the shape of dataset are not fitting")
        # 检查协方差矩阵是否为半正定阵，若否，抛出异常
        elif ~(np.all(np.linalg.eigvals(self.cov) >= 0 )):
            raise Exception('Covariance matrix is not positive define')
        else:
            # print(len(beta))
            #利用标准正态分布生成X
            X = np.random.multivariate_normal(mean = self.mean, cov = self.cov, size = self.shape[0])
            # 将全1的列向量加入到X的最后1列
            X = np.hstack((X,np.ones(shape = (self.shape[0],1))))
            # print("the size of x:",self.X.shape)
            #加入噪声生成y
            y = np.matmul(X,np.asarray(beta).reshape([-1,1])) + self.c*np.random.normal(self.noise_miu,self.noise_sigma,size = (self.shape[0],1))
            # print("the size of y:",self.y.shape)
            #合并
            dataset = pd.DataFrame(np.hstack([X,y]))
            # return 
            return dataset
        
    def cond(self,A):
        '''
        parameters:
            input:
                A: the coefficient matrix of Ax = b
        return:
            the condition number of A
        '''
        #  Do SVD decomposition on X to get singular values
        A = np.asarray(A)
        U,Sigma,_ = np.linalg.svd(A)
        # The condition number in L_2 space is equal to the largest singular value divided by the smallest singular value
        return max(Sigma)/min(Sigma)

    def correlation(self,A):
        '''
        parameters:
            input:
                A: the coefficient matrix of Ax = b
                dtype:DataFrame
        output:
            the figure of correlation matrix
        return:
            the correlation matrix of X
        '''
        # calcuate the correlation matrix of A,the last column,all values are 1,have been dropped by me.
        self.corrmat = A.corr()
        plt.subplots(dpi = 110)
        sns.heatmap(self.corrmat,vmin = -1,vmax = 1,square = True,cmap = 'RdYlBu_r',annot = True,fmt='.2f',annot_kws = {"size":8})
        plt.title('$Correlation \quad Matrix$')
        plt.xticks(list(range(A.shape[1])),A.columns)
        plt.show()
        return self.corrmat
        

    def fit(self,dataset):
        """
        parameters:
            input:dataset
                type:DataFrame
                    the last columns is label
            return : the parameters of linear model,traing_error
                type: ndarray
        """
        X = np.asarray(dataset.iloc[:,:-1]) 
        y = np.asarray(dataset.iloc[:,-1]).reshape(-1,1)
        self.alpha = np.matmul(np.linalg.inv(np.matmul(X.T,X)),np.matmul(X.T,y))
        y_predict = np.matmul(X,self.alpha)
        self.training_error = ((y - y_predict)**2).sum()
        return self.alpha,self.training_error

    def score(self,testing_data):
        """
        parameters:
            input:testing_data
                type:DataFrame
                    the last columns is label
            return : testing_error
                type: ndarray
        """
        X_test = np.asarray(testing_data.iloc[:,:-1])
        y_test = np.asarray(testing_data.iloc[:,-1]).reshape(-1,1)
        y_predict = np.matmul(X_test,self.alpha)
        self.testing_error = ((y_test - y_predict)**2).sum()
        return self.testing_error

    def predict(self,dataset):
        """
        parameters:
            input:
                dataset  d
                    type:DataFrame
                        dataset is not include labels
            return : the predict result
                type: ndarray
        """
        return np.matmul(np.asarray(dataset),self.alpha)

    def batch_gradient_descent(self,dataset,max_epoch,eta = 0.0001,show_result = False):
        """
        parameters:
            input:
                max_epoch    type:int
                eta    type:float   learning rate Default:0.001
                dataset    type:DataFrame    the last columns is label
                show_result    type:bool    Default:False
            output:
                if show_result == True:
                    output the figure of the process of training
            return : the parameters of linear model,traing_error
                type: ndarray
        """
        self.eta = eta
        X = np.asarray(dataset.iloc[:,:-1])
        y = np.asarray(dataset.iloc[:,-1]).reshape(-1,1)
        self.error_list = []
        # sample the initial params from the unifrom distribution 
        self.alpha = np.random.uniform(0,1,size = [dataset.shape[1] - 1,1])
        # print("##########################",self.alpha.shape)
        for epoch in range(max_epoch):
            error = self.score(dataset)
            self.error_list.append(error)
            self.alpha -= self.eta*np.matmul(-X.T, y - np.matmul(X,self.alpha))
        if show_result == True:
            sns.set()
            plt.figure(dpi = 110)
            plt.plot(self.error_list,c = 'b',label = '$loss \quad curve$')
            plt.title('$Batch \quad Gradient \quad Descent \quad error$')
            plt.xlabel('$epoch$')
            plt.ylabel('$error$')
            plt.legend()
            plt.show()
        return self.alpha,self.error_list
    
    def AdamOptimizer(self,dataset,max_epoch,eta = 0.001,show_result = False):
        """
        parameters:
            input:
                epsilon     type:float      the stop condition      Default:1e-6 
                eta    type:float   learning rate   Default:0.001
                dataset    type:DataFrame    the last columns is label
                show_result    type:bool    Default:False
            output:
                if show_result == True:
                    output the figure of the process of training
            return : the parameters of linear model,traing_error
                type: ndarray
        """
        beta1 = 0.9
        beta2 = 0.999
        X = np.asarray(dataset.iloc[:,:-1])
        y = np.asarray(dataset.iloc[:,-1]).reshape(-1,1)
        self.error_list = []
        # sample the initial params from the unifrom distribution 
        self.alpha = np.random.uniform(0,1,size = [dataset.shape[1] - 1,1])
        # print("##########################",self.alpha.shape)
        error = self.score(dataset)
        m = np.zeros(shape = [X.shape[1],1])
        v = np.zeros(shape = [X.shape[1],1])
        t = 1
        while t < max_epoch:
            error = self.score(dataset)
            self.error_list.append(error)
            grad = np.matmul(-X.T, y - np.matmul(X,self.alpha))
            m = beta1*m + (1 - beta1)*grad
            v = beta2*v + (1 - beta2)*np.power(grad,2)
            m_hat = m/(1 - beta1**t)
            v_hat = v/(1 - beta2**t)
            self.alpha -= eta/(np.sqrt(v_hat)+1e-8)*m_hat
            t += 1
        
        if show_result == True:
            sns.set()
            plt.figure(dpi = 110)
            plt.plot(self.error_list,c = 'b',label = '$loss \quad curve$')
            plt.title('$Adam \quad Optimizer \quad error$')
            plt.xlabel('$epoch$')
            plt.ylabel('$error$')
            plt.legend()
            plt.show()
        return self.alpha,self.error_list

    def square_R(self,dataset):
        """
        parameters:
            dataset:
                the dataset which we want to compute  R^2 above it
                dtype: DataFrame            the last column is label
            return:
                R_square
        """
        SSE = self.score(dataset)
        SSR = np.power((self.predict(dataset.iloc[:,:-1]) - dataset.iloc[:,-1].mean()),2).sum()
        SST = SSE + SSR
        #注意这里已经把全1列加入到dataset中，所以不需要减1
        R_square = 1 - SSE/SST
        return R_square
    
    def modified_square_R(self,dataset):
        """
        parameters:
            dataset:
                the dataset which we want to compute modified R^2 above it
                dtype: DataFrame            the last column is label
            return:
                modified R_square
        """
        SSE = self.score(dataset)
        SSR = np.power((self.predict(dataset.iloc[:,:-1]) - dataset.iloc[:,-1].mean()),2).sum()
        SST = SSE + SSR
        #注意这里已经把全1列加入到dataset中，所以不需要减1
        modified_R_square = 1 - (SSE/(dataset.shape[0] - dataset.iloc[:,:-1].shape[1]))/(SST/(dataset.shape[0] - 1))
        return modified_R_square


# Child Class RidgeRegression
class RidgeRegression(LinearRegression):
    '''
    RidgeRegression Object(child class):
        attribute:
            c:噪声的影响强度,取值[0,1] 默认为0
            shape:生成数据集的形状,默认为None     dtype = [sample_num,feature_num]
            mean:x所服从正态分布的均值向量       默认为None
            cov:x所服从正态分布的协方差矩阵      默认为None
            noise_miu：噪声所服从的正态分布的均值
            noise_sigma:噪声所服从的正态分布的方差
            noise默认服从标准正态分布
        function:
            generate_data(): 按照实例化对象的初始属性生成数据集
            regression_cond():计算Ridege中 X 的条件数
            regression_correlation():可视化矩阵X的相关系数矩阵
            fit():最小二乘估计训练线性模型((直接利用(X^T*X)^(-1)+CI)*X*y计算)，return alpha和RSS
            score():通过训练好的model计算测试误差
            predict():利用已训练好的model预测测试样本
            batch_gradient_descent():批量梯度下降 每一个epoch用整个训练集计算梯度
            stochastic_gradient_descent():随机梯度下降 每一个epoch中运用每一个样本计算梯度
            mini_batch_gradient_descent():小批量梯度下降 每一个epoch中运用等大小的训练集的子集计算梯度
            AdamOptimizer():Adaptive Moment Estimation
            square_R():goodness of fit
            modified_square_R():modified_square_R
            trace():Ridge coefficients as a function of the regularization
    '''
    def __init__(self,c = 0,shape = None,mean = None,cov = None,noise_miu = 0,noise_sigma = 1,C = 1):
        super(RidgeRegression,self).__init__(c,shape,mean,cov,noise_miu,noise_sigma)
        self.C = C
    
    #重写父类LinearRegression的fit方法,变为RidgeRegression
    def fit(self,dataset):
        """
        parameters:
            input:dataset
                type:DataFrame
                    the last columns is label
            return : the parameters of linear model,traing_error
                type: ndarray
        """
        X = np.asarray(dataset.iloc[:,:-1])
        y = np.asarray(dataset.iloc[:,-1]).reshape(-1,1)
        self.alpha = np.matmul(np.linalg.inv(np.matmul(X.T,X)+self.C*np.diag(np.ones(X.shape[1]))),np.matmul(X.T,y))
        y_predict = np.matmul(X,self.alpha)
        self.training_error = ((y - y_predict)**2).sum()
        return self.alpha,self.training_error

    #重写父类LinearRegression的batch_gradient_descent()方法
    def batch_gradient_descent(self,dataset,max_epoch,eta = 0.0001,show_result = False):
        """
        parameters:
            input:
                max_epoch    type:int
                eta    type:float   learning rate Default:0.0001
                dataset    type:DataFrame    the last columns is label
                show_result    type:bool    Default:False
            output:
                if show_result == True:
                    output the figure of the process of training
            return : the parameters of linear model,traing_error
                type: ndarray
        """
        self.eta = eta
        X = np.asarray(dataset.iloc[:,:-1])
        y = np.asarray(dataset.iloc[:,-1]).reshape(-1,1)
        self.error_list = []
        # sample the initial params from the unifrom distribution 
        self.alpha = np.random.uniform(0,1,size = [dataset.shape[1] - 1,1])
        # print("##########################",self.alpha.shape)
        for epoch in range(max_epoch):
            error = self.score(dataset)
            self.error_list.append(error)
            self.alpha -= self.eta*(np.matmul(-X.T, y - np.matmul(X,self.alpha)) + self.C*self.alpha)
        if show_result == True:
            sns.set()
            plt.figure()
            plt.plot(self.error_list,c = 'b',label = '$loss \quad curve$')
            plt.title('$Gradient\quad Descent \quad error$')
            plt.xlabel('$epoch$')
            plt.ylabel('$error$')
            plt.legend()
            plt.show()
        return self.alpha,self.error_list

    def AdamOptimizer(self,dataset,max_epoch,eta = 0.001,show_result = False):
        """
        parameters:
            input:
                epsilon     type:float      the stop condition      Default:1e-6 
                eta    type:float   learning rate   Default:0.001
                dataset    type:DataFrame    the last columns is label
                show_result    type:bool    Default:False
            output:
                if show_result == True:
                    output the figure of the process of training
            return : the parameters of linear model,traing_error
                type: ndarray
        """
        beta1 = 0.9
        beta2 = 0.999
        X = np.asarray(dataset.iloc[:,:-1])
        y = np.asarray(dataset.iloc[:,-1]).reshape(-1,1)
        self.error_list = []
        # sample the initial params from the unifrom distribution 
        self.alpha = np.random.uniform(0,1,size = [dataset.shape[1] - 1,1])
        # print("##########################",self.alpha.shape)
        error = self.score(dataset)
        m = np.zeros(shape = [X.shape[1],1])
        v = np.zeros(shape = [X.shape[1],1])
        t = 1
        while t < max_epoch:
            error = self.score(dataset)
            self.error_list.append(error)
            grad = np.matmul(-X.T, y - np.matmul(X,self.alpha)) + self.C*self.alpha
            m = beta1*m + (1 - beta1)*grad
            v = beta2*v + (1 - beta2)*np.power(grad,2)
            m_hat = m/(1 - beta1**t)
            v_hat = v/(1 - beta2**t)
            self.alpha -= eta/(np.sqrt(v_hat)+1e-8)*m_hat
            t += 1
        
        if show_result == True:
            sns.set()
            plt.figure(dpi = 110)
            plt.plot(self.error_list,c = 'b',label = '$loss \quad curve$')
            plt.title('$Adam\quad Optimizer \quad error$')
            plt.xlabel('$epoch$')
            plt.ylabel('$error$')
            plt.legend()
            plt.show()
        return self.alpha,self.error_list

    def trace(self,dataset,max_epoch,C_list):
        """
        parameters:
            dataset     dtype:dataframe     the last columns is label
            C_list      The list of C
            output      The figure of ridge trace
        """
        #保存原始对象的正则化参数
        init_C = self.C
        beta_list = []
        for C in C_list:
            self.C = C
            beta,_ = self.batch_gradient_descent(dataset,max_epoch)
            beta_list.append(beta)
        beta_list = np.asarray(beta_list)
        plt.figure()
        for i in range(beta_list.shape[1]):
            plt.plot(C_list,beta_list[:,i])
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Ridge coefficients as a function of the regularization')
        plt.axis('tight')
        plt.show()
        #恢复初始化参数
        self.C = init_C

#Child Class Lasso
class Lasso(LinearRegression):
    '''
    Lasso Object(child class)
    new attribute:
        C:正则化项系数 默认为1
    '''
    def __init__(self,c = 0,shape = None,mean = None,cov = None,noise_miu = 0,noise_sigma = 1,C = 1):
        super(Lasso,self).__init__(c,shape,mean,cov,noise_miu,noise_sigma)
        self.C = C
