from pandas import DataFrame
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

from XGBModel import XGBModel
from LeastSquareLoss import LeastSquareLoss

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    boston = load_boston()

    logger.info(boston.keys())

    X = DataFrame(boston["data"], columns=boston["feature_names"])
    y = boston["target"]
    # print(X)
    # print(y)

    # X['g'] = LeastSquareLoss.g(y, 0)
    # X['h'] = LeastSquareLoss.h(y, 0)
    # logger.info(X)

    xgb = XGBModel(n_estimator=5,
                   gamma=1,
                   Lambda=1,
                   max_depth=5,
                   min_samples_split=3,
                   split_threshold=100,
                   loss_func=LeastSquareLoss)

    xgb.fit(X, y, split_finding_strategy="exact")

    y_pred = xgb.predict(X)

    logger.info(xgb.score(X, y))

    plt.figure(dpi=400)
    plt.plot(y, label="true")
    plt.plot(y_pred, label="pred")
    plt.legend()
    plt.show()
