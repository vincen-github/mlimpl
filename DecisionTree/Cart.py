from matplotlib.pyplot import figure
from numpy import asarray, power as pow

import treePlotter
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class CartRegressor(object):
    """
    CartRegressor
    ================
    Training regression binary tree recursively.
    ----------------------------------------------------
    @author: vincen
    @NorthWestern University——CHINA  Mathematics faculty statistic
    @MyGithub : https://github.com/vincen-github/Machine-Learning-Code
    @MyEmail : vincen.nwu@gmail.com
    ---------------------
    Reference: <统计学习方法> 李航 P70-P71
    =====================================================-
    Attribute:
        1. max_depth: int, default=5
            Maximum depth of the tree
        2. min_samples: int, default=5
            The minimum number of samples in the leaf nodes of the tree.
        3. split_threshold: float, default=10
            the current point is split when the gini value of the optimal segmentation point is less than this value
    =========================================
    Method:
        1. fit(self, X, y, depth=1):
            The method for training cart tree.
        2. gini(self, y)
            calculate gini value.
        3. select_best_split(self, X, y)
            select the best feature and value of it in passed X and y.
        4. split_dataset(self, X, y, split_feature, split_value)
            using the feature and value passed to split the dataset.
            i distinguish continuous feature and discrete feature in this method.
        5. plot_tree(self)
            Visualize the trained tree.
        6. single_predict(self, x, tree=None)
            predict the label of single sample.
            the default value of tree is trained tree ahead.
        7. predict(self, X)
            Predict the labels of multiple samples at once.
        8. score(self, X, y)
            Computing mse of X to evaluate the model.
    =============================================================
    """

    def __init__(self, max_depth=5, min_samples=5, split_threshold=10):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.split_threshold = split_threshold

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth):
        if not isinstance(max_depth, int):
            raise TypeError
        if max_depth < 1:
            raise ValueError
        self._max_depth = max_depth

    @property
    def min_samples(self):
        return self._min_samples

    @min_samples.setter
    def min_samples(self, min_samples):
        if not isinstance(min_samples, int):
            raise TypeError
        if min_samples < 1:
            raise ValueError
        self._min_samples = min_samples

    @property
    def split_threshold(self):
        return self._split_threshold

    @split_threshold.setter
    def split_threshold(self, split_threshold):
        if not isinstance(split_threshold, (int, float)):
            raise TypeError
        if split_threshold <= 0:
            raise ValueError
        self._split_threshold = split_threshold

    def fit(self, X, y, depth=1):
        """
        fit(self, X, y)
        ===============
        Train decision trees by recursive method.
        -----------------
        Parameters:
            X: dataframe
                Current feature matrix.
            y: array_like
                Current corresponding label.
        """
        # if depth == 1, we need to initialize feature_names once
        if depth == 1:
            # get the feature name of dataset
            self.feature_names = X.columns
        # if depth is equal to max_depth u initialized.return the mean value of labels in this node
        if depth == self.max_depth:
            logger.info("The depth of the tree reaches max_depth({}).".format(self.max_depth))
            return y.mean()
            # if the numbers of samples in X less than min_samples
        # return the mean value of corresponding labels.
        if X.shape[0] < self.min_samples:
            logger.info("The number of samples in X less than min_samples({}).".format(self.min_samples))
            return y.mean()
        # get the best split point
        best_feature, best_value, gini = self.select_best_split(X, y)
        # if gini value less than  split_threshold, return the mean value
        if gini < self.split_threshold:
            logger.info("Gini value less than  split_threshold({})".format(self.split_threshold))
            return y.mean()
        # print logging
        logger.info("best_feature:{}, best_value:{}, gini:{}".format(best_feature, best_value, gini))
        # split current dataset by best_feature and best_value above
        left, right = self.split_dataset(X, y, best_feature, best_value)

        # generate the root of tree
        # @denote that u can't revise following tree to self.tree,it will cause Recursion cannot be performed.
        tree = {best_feature: {}}
        # Recursive building tree
        tree[best_feature][str(best_value) + "(l)"] = self.fit(*left, depth=depth + 1)
        tree[best_feature][str(best_value) + "(r)"] = self.fit(*right, depth=depth + 1)

        self.tree = tree
        return tree

    def gini(self, y):
        """
        Calculate the gini value of label set y
        ================================
        Parameters:
            y: label set.
        """
        # if the number of sample is less than 2, return 0 directly.
        # if not do as follows. var() will return nan in above condition.
        if y.shape[0] < 2:
            return 0
        return y.var() * y.shape[0]

    def select_best_split(self, X, y):
        """
        This method is to select the best feature under some criterion(here is gini value).
        ===============================
        Parameters:
            1. X : dataframe
                feature matrix of dataset.
            2. y : ndarray
                label vector of dataset.
        """
        gini_ls = []
        # Traverse all the features and values that can be used to split the dataset
        # we call it candidate feature and candidate value
        for candidate_feature in self.feature_names:
            # iterate candidate value of candidate feature
            for candidate_value in X[candidate_feature].drop_duplicates().sort_values():
                # split the dataset using above candidate feature and value of it.
                # denoted as left and right
                left, right = self.split_dataset(X, y, split_feature=candidate_feature, split_value=candidate_value)
                # calculate the gini value of them
                left_gini = self.gini(left[1])
                right_gini = self.gini(right[1])
                # storage candidate feature, candidate value and its performance in gini_ls with a tuple.
                gini_ls.append((candidate_feature, candidate_value, left_gini + right_gini))
        # get the best feature and value from above gini_ls using compare the 3th element to find minimal variance
        best_index = 0
        for i in range(len(gini_ls)):
            if gini_ls[i][2] < gini_ls[best_index][2]:
                best_index = i
        return gini_ls[best_index]

    def split_dataset(self, X, y, split_feature, split_value):
        """
        split the dataset passed by the split_feature and split_value
        in this method, i use the dtype of column in X passed to distinguish
        whether the feature column passed is continuous or not.
        if it is continuous. i separate the dataset with the condition that
                X[split_feature] < split_value
        and use the following condition to split dataset when the feature is discrete
                X[split_feature] == split_value
        --------------------------------------
        Parameters:
            1. X : dataframe
                feature matrix of dataset.
            2. y : ndarray
                label vector of dataset.
            3. split_feature: str
                name of feature using split the dataset.
            4. split_value: str or float or int
                value of split feature using split the dataset passed.
        """
        # if the split feature is continuous, we need to split X with the condition that feature value < split value
        if X[split_feature].dtype == "float":
            # get index of row satisfy the condition, denoted as group_index
            group_index = X[split_feature] > split_value
            return (X[~group_index], y[~group_index]), (X[group_index], y[group_index])
        # if the split feature is discrete
        # we need to separate dataset with condition that feature value = split_value
        else:
            group_index = X[split_feature] == split_value
        return (X[group_index], y[group_index]), (X[~group_index], y[~group_index])

    def plot_tree(self):
        """
        visually generated cart tree.
        """
        figure(dpi=400, figsize=(12, 12))
        treePlotter.createPlot(self.tree)

    def single_predict(self, x, tree=None):
        """
        This method can predict a single sample's label.
        """
        # If tree == None, set it as self.tree
        if tree is None:
            tree = self.tree
        # if tree is a number, it is the output we expect
        if type(tree) == float:
            return tree
        # get the feature name in the first layer
        feature_name = list(tree.keys())[0]
        # get left/right sub tree
        left, right = list(tree.get(feature_name).values())

        # Recursion Partition
        # if feature is continuous.
        if isinstance(x[feature_name], float):
            # get the split value through arbitrary branch
            split_value = list(tree.get(feature_name).keys())[0].split('(')[0]
            if x[feature_name] > float(split_value):
                return self.single_predict(x, right)
            else:
                return self.single_predict(x, left)
        # if feature is discrete
        else:
            split_value = list(tree.get(feature_name).keys())[0].split('(')[0]
            if x[feature_name] == split_value:
                return self.single_predict(x, left)
            else:
                return self.single_predict(x, right)

    def predict(self, X):
        """
        predict labels of multiple samples
        """
        res = []
        for i in range(X.shape[0]):
            res.append(self.single_predict(X.iloc[i]))

        return asarray(res)

    def score(self, X, y):
        y_pred = self.predict(X)
        return pow(y_pred - y, 2).mean()
