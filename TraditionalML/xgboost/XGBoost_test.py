from pandas import DataFrame
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

from time import time

from XGBModel import XGBModel
from LeastSquareLoss import LeastSquareLoss

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)


def time_calc(func):
    def wrapper(*args):
        start_time = time()
        res = func(*args)
        end_time = time()
        print("EXECUTION TIME:[{}]".format(end_time - start_time))
        return res

    return wrapper


@time_calc
def xgbTest(X, y):
    xgb = XGBModel(n_estimator=10,
                   gamma=1,
                   Lambda=1,
                   eta=0.3,
                   max_depth=6,
                   min_samples_split=3,
                   split_threshold=20,
                   sketch_eps=0.4,
                   loss_func=LeastSquareLoss)

    xgb.fit(X, y, split_finding_strategy="approximate")

    y_pred = xgb.predict(X)

    logger.info("MSE of xgbModel is [{}]".format(xgb.score(X, y)))

    return y_pred


if __name__ == "__main__":
    boston = load_boston()

    logger.info(boston.keys())

    X = DataFrame(boston["data"], columns=boston["feature_names"])
    y = boston["target"]

    y_pred = xgbTest(X, y)

    plt.figure(dpi=400)
    plt.plot(y, label="true")
    plt.plot(y_pred, label="pred")
    plt.legend()
    plt.show()
