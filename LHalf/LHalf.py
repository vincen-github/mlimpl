from numpy import diag, power, abs
from numpy.linalg import pinv, inv
from numpy.random import uniform
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


class LHalf(object):
    # Member variables
    _Lambda = None
    _iteration = None
    _beta = None
    _training_error = None

    def __init__(self, Lambda, iteration):
        self.Lambda = Lambda
        self.iteration = iteration

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

    y_pred = lHalf.predict(X)

    plt.figure(dpi=400)
    plt.plot(y_pred, label="y_pred")
    plt.plot(y, label="y")
    plt.legend()
    plt.show()
