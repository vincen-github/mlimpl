from numpy import diag, power, abs
from numpy.linalg import pinv, inv
from numpy.random import uniform
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class LHalf(object):
    """
    LHalf Algorithm
    ===============

    introduction of l1/2:
    ---------------------------
        l1/2 algorithm is a improved variant algorithm of lasso.
        it is a linear model as lasso but the optimization object of it is as follows
                    min loss = 1/2*||Y - Xβ|| + λ||β||_{1/2}
        Where the form of regular term ||β||_{1/2} as follows
                    ||β||_{1/2} = Σβ_i^{1/2}
    ------------------------

    How to solve this model in this class?
    ------------------------------------------
        i convert it to a known,solvable question by following approximate formula
                    β(t)_i^{1/2} ≈ β(t)_i^{2} / β(t-1)_i^{3/2}
        Where β(t) represent the iterative parameters at step t.
        Actually, the optimal point of l1/2 is similar to iterate ridge for solving lasso,
        which has explicit solution as follow
                    β* = (X^T@X + λ*diag{|β|^{-3/2})^{-1}@X@Y

    ------------------------------
    Some note of this algorithm
        1. note that this approximation becomes numerically unstable as any approaches ,
           and that this is exactly what we expect at the optimal solution.
           To avoid this problem, we can use a generalized inverse of diag{|β|^{-3/2}
        2. The optimal point of ridge is not sparse.it is different from l1/2.So we need
           to set a threshold ahead to cut off intermediate solution.We set the fault value
            of this parameter as 1 e-4.Of course u can change it to any non-negative number.
    ------------------------------

    Parameters:
        1. Lambda : λ as above mathematical formula
        2. iteration : this parameter can specify the number of ridge u want to iterate.
        3. threshold : block parameter. choose the value of cutting off threshold through specify it.
        4. beta : u can get the coefficient by call the getter method of this parameter that decorate by @property.
        5. training_error : u can get training error through this attribute instead of calling score method on training set.

    Method:
        The method name corresponds to sklearn, so i won’t repeat it here.
        1. fit()
        2. predict()
        3. score()
    ------------------------------
    """
    # Member variables
    _Lambda = None
    _iteration = None
    _threshold = None
    _beta = None
    _training_error = None

    def __init__(self, Lambda, iteration, threshold=1e-4):
        self.Lambda = Lambda
        self.iteration = iteration
        self.threshold = threshold

    @property
    def Lambda(self):
        return self._Lambda

    @Lambda.setter
    def Lambda(self, Lambda):
        if isinstance(Lambda, (int, float)):
            if Lambda >= 0:
                self._Lambda = Lambda
            else:
                raise ValueError
        else:
            raise TypeError

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, iteration):
        if isinstance(iteration, (int)):
            if iteration > 0:
                self._iteration = iteration
            else:
                raise ValueError
        else:
            raise TypeError

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        if isinstance(threshold, (float, int)):
            if threshold > 0:
                self._threshold = threshold
            else:
                raise ValueError
        else:
            raise TypeError

    @property
    def beta(self):
        return self._beta

    @property
    def training_error(self):
        return self._training_error

    def fit(self, X, y):
        nrow, ncol = X.shape
        self._beta = uniform(0, 1, size=ncol)
        logging.info("Start Training.....")
        for i in range(self._iteration):
            temp_matrix = self.Lambda * pinv(diag(power(abs(self._beta), 3 / 2)), hermitian=True)
            self._beta = inv(X.T @ X + temp_matrix) @ (X.T @ y)
        logging.info("Training completed.....")

        for i in range(ncol):
            if abs(self._beta[i]) < self.threshold:
                self._beta[i] = 0

        y_pred = self.predict(X)
        self._training_error = (y - y_pred).mean()
        logging.info("Training error:{}".format(self._training_error))

    def predict(self, X):
        return X @ self.beta

    def score(self, X, y):
        return (y - X @ self.beta).mean()


if __name__ == "__main__":
    from numpy import asarray
    import pandas as pd
    import os
    import matplotlib.pyplot as plt

    os.chdir("C:\\Users\\vincen\\Desktop")

    df = pd.read_csv(".\\input\\part-00000-aef6562b-68f9-477f-9279-a930d246f170-c000.csv", header=0, sep=",",
                     encoding="utf-8")

    data = asarray(df)

    Lambda = 1e6

    ITERATION = 1000

    X = data[:, 1:]
    y = data[:, 0]

    lHalf = LHalf(Lambda, ITERATION)

    lHalf.fit(X, y)

    print(lHalf.beta)

    y_pred = lHalf.predict(X)

    plt.figure(dpi=400)
    plt.plot(y_pred, label="y_pred")
    plt.plot(y, label="y")
    plt.legend()
    plt.show()
