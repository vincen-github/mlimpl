from Cart import CartRegressor, CartClassifier
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import os

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s -  %(asctime)s - %(filename)s -  %(name)s - %(funcName)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    os.chdir('C:\\Users\\vincen\\Desktop\\material\\Machine Learning\\data\\seaborn-data-master')
    tips = pd.read_csv("tips.csv",
                       sep=',',
                       encoding='utf-8',
                       header=0,
                       engine='python')

    X = tips.drop("tip", axis=1, inplace=False)
    y = tips["tip"]

    reg = CartRegressor(max_depth=7, min_samples_split=3, split_threshold=5)
    logger.info(reg.gini(y))

    reg.fit(X, y)

    reg.plot_tree()


    logger.info(reg.predict(X))

    logger.info(reg.score(X, y))


    plt.figure(dpi=400)
    plt.plot(reg.predict(X))
    plt.plot(y)
    plt.show()


    iris = load_iris()

    X = pd.DataFrame(iris['data'], columns=iris["feature_names"])
    y = iris['target']

    clf = CartClassifier(max_depth=7,
                         min_samples_split=5,
                         split_threshold=0.01
                         )

    clf.fit(X, y)

    score = clf.score(X, y)
    print(score)

    clf.plot_tree()

    y_pred = clf.predict(X)

    plt.figure(dpi=400)
    ax = plt.gca()
    ax.plot(y, c='red', label='true')
    ax.plot(y_pred, c='green', label='pred')
    plt.legend()
    plt.show()

