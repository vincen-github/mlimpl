from Cart import CartRegressor
import pandas as pd
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

    print(X.shape[0])

    feature_names = X.columns

    logger.info(feature_names)

    cart_regressor = CartRegressor(min_samples=3, split_threshold=5)
    logger.info(cart_regressor.gini(y))

    # print(X["time"].drop_duplicates().sort_values())

    # a, b = cart_regressor.split_dataset(X, y, "total_bill", 10)
    #
    # logger.info(a[0])
    # logger.info(b[0].shape[0])

    # cart_regressor.feature_names = X.columns
    # cart_regressor.used_features_and_values = []
    # logger.info(cart_regressor.select_best_split(X, y))
    #
    # tup = ([1, 2, 3], [4, 5, 6])
    # print(*tup)

    cart_regressor.fit(X, y)

    # cart_regressor.plot_tree()

    # for i in range(X.shape[0]):
    # print(i, cart_regressor.single_predict(X.iloc[i]))

    logger.info(cart_regressor.predict(X))

    logger.info(cart_regressor.score(X, y))

    import matplotlib.pyplot as plt

    plt.figure(dpi=400)
    plt.plot(cart_regressor.predict(X))
    plt.plot(y)
    plt.show()

