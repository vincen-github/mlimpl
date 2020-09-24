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
import treePlotter
import numpy as np
# %%
class DecisionClassifier(object):
    """
    DecisionClassifier object:
        principle:
            The model form of ClassifitionTree as following
                        y  = ΣαiI{x ∈ ci}
            where ci is the i-th leaf node, ai is the max value of P(Y|X). That is the solution of optimizal problem as following 
                            αi = argmax_{y}P(Y = y|X ∈ ci)
        parameters:
            1. max_depth: The max depth of DecisionTree
                default : 7
            2. criterion:id3 or c4.5 or gini
                feasible value:"id3" or "c4.5" or "gini"  dtype:str
            3. eps: if gain less than eps, we set current tree is single node. 
                    Dfault: 1e-2  dtype:float
        function:
            1. fit():use input data and criterion set to train a decision tree for predict new sample.
            2. __createTree():Recursively call itself to generate classification tree.
                private function of object
            3. __select_best_split_feature():Find the optimal split feature according to the set quasi-test.
                paivate function of object
            4. __calculate_purity():Calculate the purity of the data label set.
                private function of object
            5. __predict_single_sample():predict the label of input single sample.
            4. predict(): the reinforced function of above funtion, you can pass more samples you want to predict.
            5. score():compute the accuracy of input X and y.the formula of accuracy as following
                                accuracy = ΣI{\hat{y} - y}
        example:
            '''
            dataset info:
                outlook->  0: sunny | 1: overcast | 2: rain
                temperature-> 0: hot | 1: mild | 2: cool
                humidity-> 0: high | 1: normal
                windy-> 0: false | 1: true
            '''
            dataSet = pd.DataFrame([[0, 0, 0, 0, 'N'],
                            [0, 0, 0, 1, 'N'],
                            [1, 0, 0, 0, 'Y'],
                            [2, 1, 0, 0, 'Y'],
                            [2, 2, 1, 0, 'Y'],
                            [2, 2, 1, 1, 'N'],
                            [1, 2, 1, 1, 'Y']],columns = ['outlook', 'temperature', 'humidity', 'windy','label'])
            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1]
            clf = DecisionClassifier(criterion = 'c4.5')
            clf.fit(X, y, show_graph = True)
            print(clf.tree)
    """
    def __init__(self, max_depth = 7, criterion = "id3", eps = 1e-2):
        # 异常检测
        if not isinstance(max_depth, int) or not max_depth > 0:
            raise Exception("max_depth need to bigger than 0 and the type of it must be int.Please check it.")
        if criterion not in ("id3", "c4.5", "gini"):
            raise Exception("criterion must set 'id3' or 'c4.5' or 'gini'.The type of it is str.Please check it.")
        if not isinstance(eps, float):
            raise Exception("The type of eps must be float,please check it.")
        self.max_depth = max_depth
        self.criterion = criterion
        self.eps = eps

    def fit(self, X, y, show_graph = False):
        """
        fit():
            parameters:
                1. X: the feature matrix of training set.
                    dype:DataFrame
                2. y: corresponding label to training set.
                    dtype:DataFrame
                3. show_graph: if this parameter is true, the structure of the tree will be visualized.
                    dtype:bool
        """
        # 从X中获取候选特征集
        all_features = list(X.columns)
        # 获取所有feature的所有可能取值以用于后面生成决策树的分支
        self.__all_feature_dict = dict([( feature, X[feature].unique() ) for feature in all_features])
        # 建树
        self.tree = self.__createTree(X = X, y = y, candidate_features = all_features)
        
        # 若show_graph == True:
        if show_graph == True:
            treePlotter.createPlot(self.tree)
        

    def __createTree(self, X, y, candidate_features, cur_depth = 1):
        """
        __createTree():
            Recursively call itself to generate classification tree.
            private function of object
        parameters:
            1. X:The feature matrix of samples which in current leaf node.
                dtype: DataFrame
            2. y:The label matrix of smaples which in current leaf node.
                dtype : Series
            3. candidate_features:In every epoch of recursively.We need to select the best split feature from candidate_features
                dtype:list
            4. cur_depth:The depth of the current tree in each round of recursion, used for pre-pruning.
                dtype:int
        """
        """
        @@@@@注意生成分支的时候是利用整个训练集的在spilt feature上value而不是利用spilt_feature_value中的值.
        @@@@@spilt_feature_value经过分支的不断生成，落在一个叶子结点上的sample数量的减少，会有分支缺失的现象.
        基于上面的原因，我们需要将fit 与 createTree两个方法分离实现.在fit中扫描整个样本集获取特征的分支.
        """
        # 若现有树的cur_depth > max_depth,则将(X, y)中实例数最大的类作为该结点的类返回(树的预剪枝)
        if cur_depth >= self.max_depth:
            return y.iloc[np.argmax(y.value_counts())]
        #若数据集中的样本全部属于同一类别,将y中的label返回.
        if y.unique().shape[0] == 1:
            return y.iloc[0]
        #若候选属性集为空,则T为单结点树,并将(X, y)中实例数最大的类作为该结点的类返回
        if candidate_features == []:
            return y.iloc[np.argmax(y.value_counts())]
        # 否则,选择纯度增益最大的的特征,
        best_split_feature, biggest_gain = self.__select_best_split_feature(X, y, candidate_features)
        # 如果best_split_feature的信息增益小于eps,则置T为单结点数，返回(X,y)中实例数最大的类返回
        if biggest_gain < self.eps:
            return y.iloc[np.argmax(y.value_counts())]
        # 若上述情形都不满足,则使用best_split_feature划分当前递归过程的X(注意这里一定要使用self.all_feature_dict来构建分支,理由如上@@部分)
        # 初始化tree的key为best_split_feature, value为空dict,这种设置的形式导致我们可以直接进行递归
        tree = {best_split_feature: {}}
        # 使用self.all_feature_dict划分X
        for best_feature_value in self.__all_feature_dict[best_split_feature]:
            # 获取best_split_feature == best_feature_value的所有样本的index
            index = X[ X[best_split_feature] == best_feature_value ].index

            # 若其中一个结点的样本数为0，此时将该结点的output置为其父结点中(X, y)中实例数最大的类作为该结点的类.
            if y[index].shape[0] == 0:
                tree[best_split_feature][best_feature_value] = y.iloc[np.argmax(y.value_counts())]
            
            #若不为0,递归建树,同时将candidate_features浅拷贝一份,记为candidate_features_copy,
            #将best_feature从其中删除,用于下面的递归建树,这样做便可以在左右子树重复使用相同的特征划分结点。
            else:
                candidate_features_copy = candidate_features.copy()
                candidate_features_copy.remove(best_split_feature)
                tree[best_split_feature][best_feature_value] = self.__createTree(X = X.loc[index], y = y[index], candidate_features = candidate_features_copy, cur_depth = cur_depth + 1)
        return tree

    def __select_best_split_feature(self, X, y, candidate_features):
        """
        __select_best_split_feature():
            paivate function of object
            parameters:
                y:labels of samples in current leaf node 
                    dtype:Series
                candidate_features:
            return:
                1. The feature which has the biggest purity gain
                2. the biggest purity gain
        """
        # 若准则设置为id3
        if self.criterion == "id3":
            # 计算初始数据集的纯度
            init_purity = self.__calculate_purity(y)
            #purity_gain用于存储不同feature的纯度增益
            purity_gain = {}
            # 遍历X的所有特征
            for feature in candidate_features:
                # purity_fix_feature用于总和各组的purity
                purity_fix_feature = 0
                # 按feature分组
                groups = y.groupby(X[feature], axis = 0)
                # 遍历每一个组
                for name, group in groups:
                        # 计算该组的纯度后,不要忘记乘以改组样本数占y样本数的比例
                        purity_fix_feature += (group.shape[0] / y.shape[0])*self.__calculate_purity(group)
                #计算纯度增益
                purity_gain[feature] = init_purity - purity_fix_feature

            # 将纯度增益转化为Series类型
            pruity_gain = pd.Series(purity_gain)
            # 返回纯度增益最大的feature name
            return pruity_gain.index[np.argmax(pruity_gain)] , pruity_gain.max()

        # 若准则设置为c4.5
        if self.criterion == "c4.5":
            init_purity_ratio = self.__calculate_purity(y)
            # purity_ratio用于存储不同feature的纯度增益率
            purity_ratio_gain = {}
            # 遍历X的所有特征
            for feature in candidate_features:
                #purity_ratio_fix_feature用于总和各组的purity_ratio
                purity_ratio_fix_feature = 0
                # feature_purity 用于存储特征的增益
                feature_purity = 0
                #  按feature分组
                groups = y.groupby(X[feature], axis = 0)
                # 遍历每一个组
                for name, group in groups:
                    # 计算改组的纯度后,不要忘记乘以改组样本数占y样本数的比例
                    purity_ratio_fix_feature += (group.shape[0] / y.shape[0])*self.__calculate_purity(group)
                    # H(A) = -Σ(|D_k|/|D|)*(log2(|D_k|/|D|))
                    feature_purity += group.shape[0] / y.shape[0]
                # 计算纯度增益率,即信息增益除以H(A).
                # A为feature
                purity_ratio_gain[feature] = (init_purity_ratio - purity_ratio_fix_feature) / feature_purity
            
            # 将纯度增益转化为Series类型
            pruity_ratio_gain = pd.Series(purity_ratio_gain)
            # 返回纯度增益最大的feature name
            return pruity_ratio_gain.index[np.argmax(pruity_ratio_gain)] , pruity_ratio_gain.max()


    def __calculate_purity(self, y):
        """
        y ： The label set which you want to calculate the purity of it. 
            dtype: Series
        """
        # 计算各类别的概率
        prob =  y.value_counts()
        #若criterion为id3或c4.5,则计算信息熵
        if self.criterion == "id3" or "c4.5":
            # 计算信息熵
            return prob.apply(lambda x: -(x/y.shape[0])*np.log2(x/y.shape[0])).sum()

    def __predict_single_sample(self, x):
        """
        __predict_single_sample():
            This function is used to priedict unknown single sample by tree which has already trained.
            private function of object.
            parameters:
                x: the unknown sample which you want to predict.
        """
        tree = self.tree
        while isinstance(tree, dict):
            # 获取当前树的根结点特征
            feature = list(tree.keys())[0]
            # 通过特征查找下一步需要搜索的子树
            tree = list(tree.values())[0].get(x[feature])
        return tree
            
            

    def predict(self, X):
        """
        predict():
            Predict the labels of multiple samples at the same time (relying on __predict_single_sample)
            parameters:
                X: the unknown sample which you want to predict.
                    dtype:DataFrame
        """
        # label用于存储预测的标签
        predict_label = []
        for _, row in X.iterrows():
            predict_label.append(self.__predict_single_sample(row))
        return pd.Series(predict_label, index = X.index)
    
    def score(self, X, y):
        """
        score():compute the accuracy of input X and y.the formula of accuracy as following
                        accuracy = ΣI{\hat{y} - y}
        parameters:
            X : feature matrix of samples.
                dtype : DataFrame
            y : corrsponding true labels.
                dtype : Series
        """
        y_predict = self.predict(X)
        return (y_predict == y).sum() / y.shape[0]


