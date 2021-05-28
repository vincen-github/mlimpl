# author:vincen
# date:2020/4/10
# encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Parent Class
class LinearRegression(object):
    """
    LinearRegression Object(parent class):
        function:
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
    """

    def cond(self, X):
        """
        parameters:
            input:
                A: the coefficient matrix of Ax = b
        return:
            the condition number of A
        """
        #  Do SVD decomposition on X to get singular values
        A = np.asarray(X)
        U, Sigma, _ = np.linalg.svd(X)
        # The condition number in L_2 space is equal to the largest singular value
        # divided by the smallest singular value
        return max(Sigma) / min(Sigma)

    def correlation(self, A):
        """
        parameters:
            input:
                A: the coefficient matrix of Ax = b
                dtype:ndarray
        output:
            the figure of correlation matrix
        return:
            the correlation matrix of X
        """
        # calculate the correlation matrix of A,the last column,all values are 1,have been dropped by me.
        corrmat = A.corr()
        plt.subplots(dpi=110)
        sns.heatmap(corrmat, vmin=-1, vmax=1, square=True, cmap='RdYlBu_r', annot=True, fmt='.2f',
                    annot_kws={"size": 8})
        plt.title('Correlation Matrix')
        plt.xticks(list(range(A.shape[1])), A.columns)
        plt.show()
        return corrmat

    def fit(self, X, y):
        """
        parameters:
            input:X, y
                type:ndarray
                    the last columns is label
            return : the parameters of linear model,training_error
                type: ndarray
        """
        self.alpha = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
        y_predict = np.matmul(X, self.alpha)
        training_error = ((y - y_predict) ** 2).sum()
        return self.alpha, training_error

    def score(self, X, y):
        """
        parameters:
            input:X, y
                type:ndarray
                    the last columns is label
            return : testing_error
                type: ndarray
        """
        y_pred = np.dot(X, self.alpha)
        testing_error = np.power(y - y_pred, 2).mean()
        return testing_error

    def predict(self, X):
        """
        parameters:
            input:
                X
                    type:ndarray
                        X, y is not include labels
            return : the predict result
                type: ndarray
        """
        return np.matmul(np.asarray(X), self.alpha)

    def batch_gradient_descent(self, X, y, max_epoch=1000, eta=1e-4, show_result=False):
        """
        parameters:
            input:
                max_epoch    type:int
                eta    type:float   learning rate Default:0.001
                X, y    type:ndarray    the last columns is label
                show_result    type:bool    Default:False
            output:
                if show_result == True:
                    output the figure of the process of training
            return : the parameters of linear model,training_error
                type: ndarray
        """
        y = y.reshape(-1, 1)

        error_list = []
        # sample the initial params from the uniform distribution
        self.alpha = np.random.uniform(0, 1, size=[X.shape[1], 1])
        # print("##########################",self.alpha.shape)
        for epoch in range(max_epoch):
            error = self.score(X, y)
            error_list.append(error)
            self.alpha -= eta * np.matmul(-X.T, y - np.matmul(X, self.alpha))
        if show_result:
            sns.set()
            plt.figure(dpi=110)
            plt.plot(error_list, c='b', label='loss curve')
            plt.title('Batch Gradient Descent error')
            plt.xlabel('epoch')
            plt.ylabel('error')
            plt.legend()
            plt.show()
        return self.alpha, error_list

    def AdamOptimizer(self, X, y, max_epoch=1000, eta=0.001, show_result=False):
        """
        parameters:
            input:
                epsilon     type:float      the stop condition      Default:1e-6 
                eta    type:float   learning rate   Default:0.001
                X, y    type:ndarray    the last columns is label
                show_result    type:bool    Default:False
            output:
                if show_result == True:
                    output the figure of the process of training
            return : the parameters of linear model,training_error
                type: ndarray
        """
        y = y.reshape(-1, 1)

        beta1 = 0.9
        beta2 = 0.999
        error_list = []
        # sample the initial params from the uniform distribution
        self.alpha = np.random.uniform(0, 1, size=[X.shape[1], 1])
        # print("##########################",self.alpha.shape)
        error = self.score(X, y)
        m = np.zeros(shape=[X.shape[1], 1])
        v = np.zeros(shape=[X.shape[1], 1])
        t = 1
        while t < max_epoch:
            error = self.score(X, y)
            error_list.append(error)
            grad = np.matmul(-X.T, y - np.matmul(X, self.alpha))
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * np.power(grad, 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            self.alpha -= eta / (np.sqrt(v_hat) + 1e-8) * m_hat
            t += 1

        if show_result:
            sns.set()
            plt.figure(dpi=110)
            plt.plot(error_list, c='b', label='loss curve')
            plt.title('Adam Optimizer error')
            plt.xlabel('epoch')
            plt.ylabel('error')
            plt.legend()
            plt.show()
        return self.alpha, error_list

    def square_R(self, X, y):
        """
        parameters:
            X, y:
                the X, y which we want to compute  R^2 above it
                dtype: ndarray
            return:
                R_square
        """
        SSE = self.score(X, y)
        SSR = np.power((self.predict(X) - y.mean()), 2).sum()
        SST = SSE + SSR
        # 注意这里已经把全1列加入到X中，所以不需要减1
        R_square = 1 - SSE / SST
        return R_square

    def modified_square_R(self, X, y):
        """
        parameters:
            X, y:
                the X, y which we want to compute modified R^2 above it
                dtype: ndarray            the last column is label
            return:
                modified R_square
        """
        SSE = self.score(X, y)
        SSR = np.power((self.predict(X) - y.mean()), 2).sum()
        SST = SSE + SSR
        # 注意这里已经把全1列加入到X, y中，所以不需要减1
        modified_R_square = 1 - (SSE / (X.shape[0] - X.shape[1] - 1)) / (
                SST / X.shape[0])
        return modified_R_square


# Child Class RidgeRegression
class RidgeRegression(LinearRegression):
    """"
    RidgeRegression Object(child class):
        function:
            regression_cond():计算Ridge中 X 的条件数
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
    """

    def __init__(self, C=1):
        self.C = C

    # 重写父类LinearRegression的fit方法,变为RidgeRegression
    def fit(self, X, y):
        """
        parameters:
            input:X, y
                type:ndarray
                    the last columns is label
            return : the parameters of linear model,training_error
                type: ndarray
        """
        self.alpha = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.C * np.diag(np.ones(X.shape[1]))),
                               np.matmul(X.T, y))
        y_predict = np.matmul(X, self.alpha)
        training_error = ((y.reshape(y_predict.shape) - y_predict) ** 2).sum()
        return self.alpha, training_error

    # 重写父类LinearRegression的batch_gradient_descent()方法
    def batch_gradient_descent(self, X, y, max_epoch=1000, eta=1e-4, show_result=False):
        """
        parameters:
            input:
                max_epoch    type:int
                eta    type:float   learning rate Default:0.0001
                X, y    type:ndarray    the last columns is label
                show_result    type:bool    Default:False
            output:
                if show_result == True:
                    output the figure of the process of training
            return : the parameters of linear model,training_error
                type: ndarray
        """
        y = y.reshape(-1, 1)
        error_list = []
        # sample the initial params from the uniform distribution
        self.alpha = np.random.uniform(0, 1, size=[X.shape[1], 1])
        # print("##########################",self.alpha.shape)
        for epoch in range(max_epoch):
            error = self.score(X, y)
            error_list.append(error)
            self.alpha -= eta * (np.matmul(-X.T, y - np.matmul(X, self.alpha)) + self.C * self.alpha)
        if show_result:
            sns.set()
            plt.figure()
            plt.plot(error_list, c='b', label='loss  curve')
            plt.title('Gradient Descent  error')
            plt.xlabel('epoch')
            plt.ylabel('error')
            plt.legend()
            plt.show()
        return self.alpha, error_list

    def AdamOptimizer(self, X, y, max_epoch=1000, eta=0.001, show_result=False):
        """
        parameters:
            input:
                epsilon     type:float      the stop condition      Default:1e-6 
                eta    type:float   learning rate   Default:0.001
                X, y    type:ndarray    the last columns is label
                show_result    type:bool    Default:False
            output:
                if show_result == True:
                    output the figure of the process of training
            return : the parameters of linear model,training_error
                type: ndarray
        """
        y = y.reshape(-1, 1)
        beta1 = 0.9
        beta2 = 0.999
        error_list = []
        # sample the initial params from the uniform distribution
        self.alpha = np.random.uniform(0, 1, size=[X.shape[1], 1])
        # print("##########################",self.alpha.shape)
        error = self.score(X, y)
        m = np.zeros(shape=[X.shape[1], 1])
        v = np.zeros(shape=[X.shape[1], 1])
        t = 1
        while t < max_epoch:
            error = self.score(X, y)
            error_list.append(error)
            grad = np.matmul(-X.T, y - np.matmul(X, self.alpha)) + self.C * self.alpha
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * np.power(grad, 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            self.alpha -= eta / (np.sqrt(v_hat) + 1e-8) * m_hat
            t += 1

        if show_result:
            sns.set()
            plt.figure(dpi=110)
            plt.plot(error_list, c='b', label='loss  curve')
            plt.title('Adam Optimizer  error')
            plt.xlabel('epoch')
            plt.ylabel('error')
            plt.legend()
            plt.show()
        return self.alpha, error_list

    def trace(self, X, y, max_epoch, C_list):
        """
        parameters:
            X, y     dtype:ndarray
            C_list      The list of C
            output      The figure of ridge trace
        """
        # 保存原始对象的正则化参数
        init_C = self.C
        beta_list = []
        for C in C_list:
            self.C = C
            beta, _ = self.batch_gradient_descent(X, y, max_epoch)
            beta_list.append(beta)
        beta_list = np.asarray(beta_list)
        plt.figure()
        for i in range(beta_list.shape[1]):
            plt.plot(C_list, beta_list[:, i])
        plt.xlabel('alpha')
        plt.ylabel('weights')
        plt.title('Ridge coefficients as a function of the regularization')
        plt.axis('tight')
        plt.show()
        # 恢复初始化参数
        self.C = init_C


# Child Class Lasso
class Lasso(LinearRegression):
    """
    Lasso Object(child class)
        attribute:
            C:正则化项系数 默认为1
        function:
            regression_cond():计算 X 的条件数
            regression_correlation():可视化矩阵X的相关系数矩阵
            score():通过训练好的model计算测试误差
            predict():利用已训练好的model预测测试样本
            square_R():goodness of fit
            modified_square_R():modified_square_R
            iterate_ridge:Solve lasso by iterative method of solving ridge multiple times
            coordinate_descent():solve lasso by coordinate descent
    """

    def __init__(self, C=1):
        self.C = C

    def iterate_ridge(self, X, y, max_epoch, tol=1e-4):
        """
        iterate_ridge:
            Solve lasso by iterative method of solving ridge multiple times
            Iterated Ridge Regression using the approximation
                            |w| =~ norm(w,2)/norm(w,1)
        parameters:
            input:
                1.X, y      type:ndarray
                    notation:the last columns is label and the penultimate column is a vector of all ones,but if your label y has been standardized,you can drop the penultimate column.
                2.max_epoch     type:int
                    notation:The max number of iterate.
                3.tol       type:float
                    notation:Threshold with coefficient set to 0. The reason why this parameter is necessary is because RidgeRegression does not have the sparsity.So if we do not set this parameter.The coefficients will close to zero but not equal to it.
            return: lasso estimation result
        """
        # sample the initial params from the uniform distribution
        self.alpha = np.random.uniform(0, 1, size=X.shape[1])
        for epoch in range(max_epoch):
            temp_matrix = self.C * np.linalg.pinv(np.diag(np.abs(self.alpha)), hermitian=True)
            self.alpha = np.matmul(np.linalg.inv(np.dot(X.T, X) + temp_matrix), np.matmul(X.T, y))
        # 置0时，要避开偏置项
        for i in range(self.alpha.shape[0] - 1):
            if self.alpha[i] < tol:
                self.alpha[i] = 0
        return self.alpha

    def coordinate_descent(self, X, y, eps=1e-6):
        """
        coordinate_descent:
            Solve lasso by coordinate descent.
            parameters:
            input:
                1.X, y      type:ndarray
                    notation:the last columns is label and the penultimate column is a vector of all ones,but if your label y has been standardized,you can drop the penultimate column.
                2.eps           type:float      default:1e-6
                    notation:if ||alpha_t - alpha_{t+1} || < eps,we think it is convergence
            return
                lasso estimation result
        """
        n = X.shape[0]
        # 初始化alpha
        self.alpha = np.random.uniform(0, 1, size=X.shape[1])
        # 声明变量，用于浅拷贝alpha，以便于对比前一次的参数与后一次的参数，使用ones的原因是保证第一次可以进入循环
        alpha_copy = -np.ones(X.shape[1])
        # 前一次迭代的参数与后一次迭代的参数差的范数小于eps,则终止迭代
        while np.linalg.norm(alpha_copy - self.alpha) > eps:
            # 浅拷贝alpha
            alpha_copy = self.alpha.copy()
            for l in range(X.shape[1]):
                # 取出第l列
                X_l = X[:, l]
                # 将去除第l列的X记为X_drop
                X_drop = np.delete(X, l, axis=1)
                # 将去除第l列的alpha记为alpha_drop
                alpha_drop = np.delete(self.alpha, obj=l)
                # 构建去除第l列后的残差向量
                r = y - np.dot(X_drop, alpha_drop).reshape(y.shape)
                # 构建与C比较的量
                compare = (2 / n) * np.dot(X_l.T, r)
                if compare > self.C:
                    self.alpha[l] = n * (compare - self.C) / (2 * (X_l ** 2).sum())
                elif compare < -self.C:
                    self.alpha[l] = n * (compare + self.C) / (2 * (X_l ** 2).sum())
                else:
                    self.alpha[l] = 0
        return self.alpha
