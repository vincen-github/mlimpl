# import the system python package instead of the package that in same directory as this file and package name imported.
from __future__ import absolute_import

from numpy import zeros

from BaseTree import BaseTree
from LeastSquareLoss import LeastSquareLoss
from LossFunc import LossFunc

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBModel(object):
    """
    XGBModel Object:
    ===========
    xgb is a class that implement a scalable tree boosting system proposed by TianQi Chen.
    @author: vincen
    @NorthWestern University——CHINA  Mathematics faculty statistic
    @MyGithub : https://github.com/vincen-github/Machine-Learning-Code
    @MyEmail : vincen.nwu@gmail.com
    ----------------------------
    Reference
    ==============
        XGBoost: A Scalable Tree Boosting System, TianQi Chen, University of Washington
    ----------------------------
    Attribute:
    ==========
        1. n_estimator : int, default=100
            the number of ensemble base model
            Default:100
        2. gamma : float, default=1
            The parameters of regularization term ———— gamma*T.
            where T is the number of the child leaves.
        3. Lambda : float, default = 1
            The parameters of regularization term ———— Lambda*(||w||^2)/2
        4. max_depth : int, default = 3
            the max depth of every base tree
        5. min_samples_split: int, default = 3
            The minimum number pf samples required to split and internal node
            consider `min_samples_split` as the minimum number.
        6. split_threshold: int or float, default=10
            The current point is split when the gain value of the optimal segmentation point is less than this value
        7. loss_func: implement of abstract class., default=LeastSquareLoss
            loss function that u can specify in xgboost.
            u need to implement the abstract class named loos_func before construct XGBModel object.
        8. sketch_eps: float, range=(0,1], default=0.1
            we call sketch_eps is approximation factor.we use it to choose splitting candidates by following metric.
                |r_k(s_{k,j}) - r_k(s_{k,j+1)| < sketch_eps
            where r_k:R→[0,∞] is the rank function defined on the paper of TianQi Chen.
    ------------------------------------------------------------
    Method:
    ===========
        The data u passed to following method must be digitize.
        1. fit(X, y, split_strategy = "exact"):
            split_strategy: u can choose "exact" or "approximate" to specify how to
             choose the split feature and its value of it in process of training.
        2, predict(X):
            Predict the label of new samples X.
        3. score(X, y):

    """

    def __init__(self,
                 n_estimator=100,
                 gamma=1,
                 Lambda=1,
                 max_depth=3,
                 min_samples_split=3,
                 split_threshold=10,
                 loss_func=LeastSquareLoss):

        self.n_estimator = n_estimator
        self.gamma = gamma
        self.Lambda = Lambda
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.split_threshold = split_threshold
        self.loss_func = loss_func

    @property
    def n_estimator(self):
        return self._n_estimator

    @n_estimator.setter
    def n_estimator(self, n_estimator):
        if not isinstance(n_estimator, int):
            raise TypeError
        if n_estimator < 0:
            raise ValueError
        self._n_estimator = n_estimator

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        if not isinstance(gamma, (int, float)):
            raise TypeError
        if gamma < 0:
            raise ValueError
        self._gamma = gamma

    @property
    def Lambda(self):
        return self._Lambda

    @Lambda.setter
    def Lambda(self, Lambda):
        if not isinstance(Lambda, (int, float)):
            raise TypeError
        if Lambda < 0:
            raise ValueError
        self._Lambda = Lambda

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
    def min_samples_split(self):
        return self._min_samples_split

    @min_samples_split.setter
    def min_samples_split(self, min_samples_split):
        if not isinstance(min_samples_split, int):
            raise TypeError
        if min_samples_split < 1:
            raise ValueError
        self._min_samples_split = min_samples_split

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

    @property
    def loss_func(self):
        return self._loss_func

    @loss_func.setter
    def loss_func(self, loss_func):
        if not issubclass(loss_func, LossFunc):
            raise TypeError
        self._loss_func = loss_func

    def fit(self, X, y):
        """
        fit(self, X, y)
        --------------
        Parameters:
            X: DataFrame.
            y: array_like
        """
        # ensemble list is used to store multiple trained base models.
        self.ensemble = []
        # Train the base model one by one
        for k in range(self.n_estimator):
            logger.info("ITERATION NUMBER: {}".format(k))
            # calculate y_hat using the base model already exists.
            y_hat = self.predict(X)
            # calculate g and h for every samples in dataset based on the based model has already obtained.
            # and set them as two columns in X.
            X['g'] = LeastSquareLoss.g(y, y_hat)
            X['h'] = LeastSquareLoss.h(y, y_hat)
            # train the base model in current iteration.
            base_model = BaseTree(max_depth=self.max_depth,
                                  min_samples_split=self.min_samples_split,
                                  split_threshold=self.split_threshold,
                                  gamma=self.gamma,
                                  Lambda=self.Lambda,
                                  loss_func=self.loss_func)
            base_model.train(X, y)
            # append base_model to ensemble
            self.ensemble.append(base_model)

    def predict(self, X):
        """
        predict()
        parameters:
            X:the dataset you want to predict.
                type:array_like
        """
        # if the length of self.base_model is 0.return 0 directly.
        if len(self.ensemble) == 0:
            return zeros(X.shape[0])
        # otherwise, return the sum of output of every base model.
        res = 0
        for base_model in self.ensemble:
            res += base_model.predict(X)
        return res
