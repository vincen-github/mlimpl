from math import ceil
from numpy import sqrt, unique, argmax, asarray
from numpy.random import randint, seed, choice

from BaseTree import CartClassifier

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class RandomForestClassifier(object):
    """
    A random forest classifier.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to build
    each tree.

    Parameters
    ----------
    1. n_estimators: int, default=100
        The number of trees in the forest.

    2. max_depth: int, default=5
        The maximum depth of the tree.

    3. min_samples_split: int, default=2
        The minimum number of samples required to split an internal node.

    4. split_threshold: float, range=(0,1], default=0.1
        value of split feature using split the dataset passed.

    5. max_features: "sqrt", int, default='sqrt'
        The number of features to consider when looking for the best split:

        - If int, then consider 'max_features' features at each split.
        - If "sqrt", then 'max_features=sqrt(n_features)'

    6.bootstrap: bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    7. max_samples : int or float ,default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus
          `max_samples` should be in the interval `(0, 1)`.

    7. random_state : int, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider then looking for the best spit at each node.

    ---------------------------------------------------------------
    Example 1:
        from matplotlib import pyplot as plt
        from rf import RandomForestClassifier
        import pandas as pd
        from sklearn.datasets import load_iris

        iris = load_iris()
        X = pd.DataFrame(iris['data'], columns=iris["feature_names"])
        y = iris['target']
        clf = RandomForestClassifier(n_estimator=100,
                                     max_depth=7,
                                     min_samples_split=5,
                                     split_threshold=0.1,
                                     max_features='sqrt',
                                     bootstrap=True,
                                     max_samples=0.7,
                                     random_state=42
                                     )
        clf.fit(X, y)
        score = clf.score(X, y)
        print(score)
        y_pred = clf.predict(X)

    Example 2:
        from rf import RandomForestClassifier
        import pandas as pd
        import os

        clf = RandomForestClassifier(n_estimator=100,
                                     max_depth=7,
                                     min_samples_split=5,
                                     split_threshold=0.1,
                                     max_features='sqrt',
                                     bootstrap=True,
                                     max_samples=0.7,
                                     random_state=42
                                     )
        os.chdir("C:\\Users\\vincen\\Desktop\\material\\Machine Learning\\data")
        # 注意index从1开始
        loan = pd.read_csv("贷款申请样本数据表.csv", header=0, sep=',', encoding='gbk')
        X = loan.iloc[:, :-1]
        y = loan.iloc[:, -1]
        clf.fit(X, y)
        y_pred = clf.predict(X)
        score = clf.score(X, y)
        print(score)
    """

    def __init__(self,
                 n_estimator=100,
                 max_depth=5,
                 min_samples_split=2,
                 split_threshold=0.1,
                 max_features='sqrt',
                 bootstrap=True,
                 max_samples=True,
                 random_state=None
                 ):
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.split_threshold = split_threshold
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.random_state = random_state

        self.ensemble = []

    def fit(self, X, y):

        y = asarray(y)

        nrow, ncol = X.shape

        seed(self.random_state)

        if self.max_features == 'sqrt':
            self.max_features = ceil(sqrt(ncol))

        for k in range(self.n_estimator):
            row_samples_num = nrow
            # sub-row-sampling
            if self.bootstrap:
                if isinstance(self.max_samples, float):
                    row_samples_num = ceil(self.max_samples * nrow)
                elif isinstance(self.max_samples, int):
                    row_samples_num = self.max_samples

                row_samples_index = choice(a=list(range(nrow)),
                                           size=row_samples_num,
                                           replace=True
                                           )

            else:
                row_samples_index = list(range(nrow))

            # sub-col-sampling
            col_samples_index = choice(a=list(range(ncol)),
                                       size=self.max_features,
                                       replace=False
                                       )

            subX = X.iloc[row_samples_index, col_samples_index]
            suby = y[row_samples_index]

            # build base model
            clf = CartClassifier(max_depth=self.max_depth,
                                 min_samples_split=self.min_samples_split,
                                 split_threshold=self.split_threshold)
            clf.fit(subX, suby)

            logger.info("The training error of the {}-th tree is {}".format(k, clf.score(X, y)))

            self.ensemble.append(clf)

    def single_predict(self, x):
        res = []
        for estimator in self.ensemble:
            res.append(estimator.single_predict(x))
        labels, counts = unique(res, return_counts=True)
        return labels[argmax(counts)]

    def predict(self, X):
        res = []
        for i in range(X.shape[0]):
            res.append(self.single_predict(X.iloc[i]))
        return asarray(res)

    def score(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).sum() / y.shape[0]
