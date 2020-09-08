"@author:vincen"
import pandas as pd
import numpy as np

class MultinomialNB(object):
    """
    MultionmialNB object:
        model principle:
            posterior ∝ likeihood*priori
            in the following equation.the likeihood P(X(j) = x(j)|y = c_k) and prior P(y = c_k) both obey the multinomial distribution
            P(y = c_k|X = x) ∝ P(X = x|y = c_k)P(y = c_k) = ΠP(X(j) = x(j)|y = c_k)P(y = c_k)
        attribute of instance object:
            1.laplacian_smoothing: If you want to use the laplacian_smoothing,you need to keep the parameter as True 
                Defalut:Ture
            2.lambd: the parameters of laplacian smoothing 
                detype:float
                Defalut: 1
        funtion:
            fit():build model by input data.
            predict_single_instance():use this function to predict single sample. 
                private function of instance object.we can't call this function outside this class
            predict(): predict sample by trained model.
            score():set accuracy = ΣI(y_predict == ytest) as criterion to measure the goodness of trained model.
    """
    def __init__(self, laplacian_smoothing = True, lambd = 1):
        self.laplacian_smoothing = laplacian_smoothing
        self.lambd = lambd
        #异常检测
        if self.lambd < 0:
            raise Exception("lambd must be Greater than or equal to 0")
        if not isinstance(lambd, (int,float)):
            raise Exception("please check the type of lambd, this parameter's type need to set as int or float.")
        #若laplacian_smoothing设置为False，模型中不引入laplace平滑等价于lambda = 0
        if self.laplacian_smoothing == False:
            lambd = 0

    def fit(self, X, y):
        """
        fit():
        parameters:
            1.X: Sampling of input space 
                dtype:DataFrame
            2.y: Corresponding sample label  
                dtype:DataFrame
            3.Xtest: the sample you want to predict,the Defalut value is None.if it's defalut value is keeped.We will set it as X.
                dtype:None or DataFrame
        """
        #获取dataset的特征数与样本数
        nrow =  X.shape[0]
        ncol = X.shape[1]
        #获取dataset中的label的种类
        self.label_names = y.unique()
        self.feature_names = X.columns
        #按照label对X进行分组
        Xgrouped =  X.groupby(y)
        # priori用于存储priori,即 P(y = c_k),类型为dict
        self.priori = {}
        """
        likeihood用于存储似然,即P(X(j) = x(j)|y = c_k)
        类型为dict类型,key为label_name,value为dict类型
        第二层的字典key值为feature_name,value为value_counts(Series类型)
        """
        self.likeihood = {}
        print("正在训练多项分布朴素贝叶斯...")
        #固定label_value
        for label_name, group in Xgrouped:
            #计算priori,即统计各个组中样本的个数
            self.priori[label_name] =  (group.shape[0] + self.lambd) / (nrow + ncol * self.lambd)
            #固定feature,计算likeihood
            likeihood_fix_label = {}
            for feature_name in self.feature_names:
                #在计算似然时,加入laplace平滑项
                likeihood_fix_label[feature_name] = (group[feature_name].value_counts() + self.lambd) / (group.shape[0] + self.lambd * group[feature_name].unique().shape[0])
            self.likeihood[label_name] = likeihood_fix_label
        print("训练完成.")

    def __predict_single_instance(self, instance):
        """
        principle:
            P(y = c_k|X = x) ∝ P(X = x|y = c_k)P(y = c_k) = ΠP(X(j) = x(j)|y = c_k)P(y = c_k)
        parameters:
            instance:a sample you want to predict
            dtype:Series
        """
        #posterior_dict用于存放不同c_k下priori*likeihood的值
        posterior_dict = {}
        #固定principle中的c_k
        for label_name in self.label_names:
            #因posterior_list需要连乘,ΠP(X(j) = x(j)|y = c_k)P(y = c_k),故初始化为priori,下面只需要连乘P(X(j) = x(j)|y = c_k)
            posterior = self.priori[label_name]
            #拿到fit中所得到的固定label_name的所有feature似然信息
            fix_label_likeihood_dict = self.likeihood[label_name]
            #固定feature_name(即固定X(j))
            for feature_name in instance.index:
                try:
                    #通过instance的feature_name对应的feature_value得到对应的P(X(j) = x(j)|y = c_k),再连乘
                    posterior *= fix_label_likeihood_dict[feature_name].loc[instance[feature_name]]
                except KeyError:
                    #如果出现KeyError,说明测试样本中feature值没有出现在训练集中，这时候要将未出现的feature_value加入self.likeihood中
                    self.likeihood[label_name][feature_name].loc[instance[feature_name]] = self.lambd / (self.lambd * self.likeihood[label_name][feature_name].unique().shape[0])
                    """
                    加入featrue新的取值时,会出现该feature_name下的Σlikeihood ≠ 1,这是因为likeihood的计算式分母中laplace平滑参数是与该feature可取值的个数Sj是有关的
                    但likeihood的计算式中,分母是一样的，这意味着我们只需要归一化该feature的likeihood便可以解决该问题
                    """
                    #归一化
                    self.likeihood[label_name][feature_name] /= self.likeihood[label_name][feature_name].sum() 
                    #设置fix_label_likeihood_dict为更新后的dict
                    fix_label_likeihood_dict = self.likeihood[label_name]
                    #进行连乘
                    posterior *= fix_label_likeihood_dict[feature_name].loc[instance[feature_name]]

            #将posterior放入字典
            posterior_dict[label_name] = posterior
        #返回字典中value较大的key作为predict result
        max_posterior = max(posterior_dict.values())
        for key, value in posterior_dict.items():
            if value == max_posterior:
                return key
                

    def predict(self, X):
        """
        parameters:
            X:the sample you want to predict
            dtype:DataFrame
        """
        y_predict = []
        for _, row in X.iterrows():
            y_predict.append(self.__predict_single_instance(row))
        return pd.Series(y_predict)


    def score(self, X, y):
        """
        accuracy = ΣI(y_predict == ytest)
        parameters:
            X:the feature matrix of samples
            y: the true label of samples
        """
        y_predict = list(self.predict(X))
        y = list(y)
        return sum([1  for i in range(len(y)) if y[i] == y_predict[i]]) / len(y)

class GaussianNB(object):
        """
        GaussianNB object:
            model principle:
                posterior ∝ likeihood*priori
                in the following equation.the likeihood P(X(j) = x(j)|y = c_k) and prior P(y = c_k) both obey the gaussian distribution
                P(y = c_k|X = x) ∝ P(X = x|y = c_k)P(y = c_k) = ΠP(X(j) = x(j)|y = c_k)P(y = c_k)
            funtion:
                fit():build model by input data.
                __predict_single_instance():use this function to predict single sample. 
                    private function of instance object.we can't call this function outside this class
                predict(): predict sample by trained model.
                score():set accuracy = ΣI(y_predict == ytest) as criterion to measure the goodness of trained model.
        """
        def __normal_density(self, x, mu = 0, sigma = 1):
            """
            normal_distribution():
                f(x) = 1/√2Πσ exp(-(x-μ)²/(2σ²))
                private function of instance object.we can't call this function outside the class.
            parameters:
                x:the independent variable of  the function f(x)
                    dtype:scaler(int or float)
                mu: the mean value of normal_distribution.
                    dtype: int or float
                sigma:the value of sqrt variance,we also call it std.
                    dtype: int or float
                    we need to restrict the value of sigma is not lesser than zero.
            """
            if not isinstance(sigma, (float, int)) or not isinstance(mu, (float, int)):
                raise Exception("The type of sigma and mu must be float or int.")
            if sigma < 0:
                raise Exception("We restrict the value of sigma is not lesser than zero.")
            return np.exp(-(x - mu)**2 / (2*sigma**2)) / (np.sqrt(2*np.pi)*sigma)


        def fit(self, X, y):
            """
                parameters:
                    1.X: Sampling of input space.
                        dtype:DataFrame
                    2.y: Corresponding sample label.  
                        dtype:DataFrame
                    3.Xtest: the sample you want to predict,the Defalut value is None.if it's defalut value is keeped.We will set it as X.
                        dtype:None or DataFrame
            """
            #获取样本量与特征维数
            nrow = X.shape[0]
            ncol = X.shape[1]
            #获取feature_name
            self.feature_names = X.columns
            #获取label_name
            self.label_names = y.unique()
            Xgrouped = X.groupby(y)
            #构建dict,存储prior
            self.priori = {}
            #构建mean,std的dict,存储各groupmean,std作为正态分布的参数
            self.mean_dict, self.std_dict = {}, {}
            for name, group in Xgrouped:
                self.priori[name] = group.shape[0] / nrow
                self.mean_dict[name] = group.mean()
                self.std_dict[name] = group.std()
            
        def __predict_single_instance(self, instance):
            """
            principle:
                P(y = c_k|X = x) ∝ P(X = x|y = c_k)P(y = c_k) = ΠP(X(j) = x(j)|y = c_k)P(y = c_k)
            parameters:
                instance:a sample you want to predict
                dtype:Series
            """
            #posterior_dict用于存储priori*likeihood
            posterior_dict = {}
            #固定label_name
            for label_name in self.label_names:
                #因posterior_list需要连乘,ΠP(X(j) = x(j)|y = c_k)P(y = c_k),故初始化为priori,下面只需要连乘P(X(j) = x(j)|y = c_k)
                posterior = self.priori[label_name]
                #下计算likeihood
                #固定feature_name
                for feature_name in self.feature_names:
                    mu = self.mean_dict[label_name].loc[feature_name]
                    sigma = self.std_dict[label_name].loc[feature_name]
                    posterior *= self.__normal_density(instance[feature_name],mu = mu, sigma = sigma)
                posterior_dict[label_name] = posterior
            #返回字典中value较大的key作为predict result
            max_posterior = max(posterior_dict.values())
            for key, value in posterior_dict.items():
                if value == max_posterior:
                    return key
        
        def predict(self, X):
            """
            parameters:
                X:the sample you want to predict
                dtype:DataFrame
            """
            y_predict = []
            for _, row in X.iterrows():
                y_predict.append(self.__predict_single_instance(row))
            return pd.Series(y_predict)
            
        def score(self, X, y):
            """
            accuracy = ΣI(y_predict == ytest)
            parameters:
                X:the feature matrix of samples
                y: the true label of samples
            """
            y_predict = list(self.predict(X))
            y = list(y)
            return sum([1  for i in range(len(y)) if y[i] == y_predict[i]]) / len(y)        

