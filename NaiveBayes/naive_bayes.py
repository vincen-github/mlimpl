"autor:vincen"
import pandas as pd

class NaiveBayes(object):
    """
    NativeBayes Object:
        function:
            1.MultinomialNB:Multinomial Naive Bayes
            2.score_by_accuracy:ΣI(y_predict == ytest)
    """
    def __init__(self):
        pass

    def MultinomialNB(self, X, y, Xtest = None, laplacian_smoothing = True,lambd = 1):
        """
        MultionmialNB:
            model:
                posterior ∝ likeihood*priori
                in the following equation.the likeihood P(X(j) = x(j)|y = c_k) and prior P(y = c_k) both obey the multinomial distribution
                P(y = c_k|X = x) ∝ P(X = x|y = c_k)P(y = c_k) = ΠP(X(j) = x(j)|y = c_k)P(y = c_k)
        parameters:
            1.X: Sampling of input space 
                dtype:DataFrame
            2.y: Corresponding sample label  
                dtype:DataFrame
            3.Xtest: the sample you want to predict,the Defalut value is None.if it's defalut value is keeped.We will set it as X.
                dtype:None or DataFrame
            4.laplacian_smoothing: If you want to use the laplacian_smoothing,you need to keep the parameter as True 
                Defalut:Ture
            5.lambd: the parameters of laplacian smoothing 
                detype:float
                Defalut: 1
        """
        if lambd < 0:
            raise Exception("lambd must be Greater than or equal to 0")
        if not isinstance(lambd, (int,float)):
            raise Exception("please check the type of lambd, this parameter's type need to set as int or float.")
        #若laplacian_smoothing设置为False，模型中不引入laplace平滑等价于lambda = 0
        if laplacian_smoothing == False:
            lambd = 0
        if type(Xtest) == type(None):
            Xtest = X
        nrow =  X.shape[0]
        ncol = X.shape[1]
        #统计label的类别数与每种类别下有多少样本，在P(y = c_k|X = x) = P(X = x|y = c_k)P(y = c_k) = ΠP(X(j) = x(j)|y = c_k)P(y = c_k)的计算中,连乘号后验概率无关,我们对于每一个样本与每一个c_k，先去计算后验，再去循环计算似然。
        label_value_counts = y.value_counts()
        #统计X各个特征的类别数与每种类别下有多少样本
        feature_value_counts_list = [X.iloc[:, j].value_counts() for j in range(ncol)]
        #获取特征名称
        self.feature_names = [item.name for item in feature_value_counts_list]
        #y_predict用于存储预测值
        self.y_predict = []
        #遍历需要predict的样本集(先解决一个样本的预测问题)
        for l in range(Xtest.shape[0]):
            #固定测试样本
            predict_sample = Xtest.iloc[l]
            #posterior存储每一个样本的likeihood*prioir
            posterior = []
            #对于不同的c_k(c_k为label的取值)计算prioir,label代表标签值
            for label in label_value_counts.index:
                #number代表该label在training_data对应的总样本数
                same_label_sample_number = label_value_counts.loc[label]
                #计算属于各类别的先验概率
                priori = (same_label_sample_number + lambd) / (nrow + label_value_counts.shape[0]*lambd)
                #************************************************************************
                #下面计算likeihood
                #likeihood是连乘形式，故初始化为1
                likeihood = 1
                #ΠP(X(j) = x(j)|y = c_k)中需要将X的分量提取出来,同时计算似然的分子分母部分
                for j in range(ncol):
                    # x_j代表predict_sample的第j个分量
                    x_j = predict_sample.iloc[j]
                    # 计算laplace平滑部分时，需要统计第j列特征的类别数
                    S_j = feature_value_counts_list[j].shape[0]
                    #遍历所有的样本,统计X中满足 X_i^j = x^j and y_i = label的样本个数,初始时设置分子,分母为0进行累加
                    molecular,denominator = 0,0
                    for i in range(nrow):
                        if y.iloc[i] == label:
                            denominator += 1
                            if X.iloc[i,j] == x_j:
                                molecular += 1
                    #计算带laplace平滑的似然
                    likeihood *= (molecular + lambd) / (denominator + S_j*lambd) 
                #posterior ∝ likeihood*priori
                posterior.append(likeihood*priori)
            self.y_predict.append(label_value_counts.index[posterior.index(max(posterior))])
        return self.y_predict

    def score_by_accuracy(self, ytest):
        """
            accuracy = ΣI(y_predict == ytest)
            parameters:
                ytest: the true label of samples
        """
        ytest = list(ytest)
        return sum([1 for item in zip(ytest,self.y_predict) if item[0] == item[1]])/len(ytest)



