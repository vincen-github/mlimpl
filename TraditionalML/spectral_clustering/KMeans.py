import numpy as np


class KMeans(object):
    """
    KMeans
    ------
        A heuristic algorithm for cluster

    Available attributes
    ---------------------
        1. k: int
            The number of class.
        
        2. init : {'k-means++', 'random'}, default = 'k-means++'
            Method for initialization:

            'k-means++' : selects initial cluster centers for k-mean
            clustering in a smart way to speed up convergence. See section
            Notes in k_init for more details.

            'random': choose `n_clusters` observations (rows) at random from data
            for the initial centroids.

            If an ndarray is passed, it should be of shape (n_clusters, n_features)
            and gives the initial centers.

            If a callable is passed, it should take arguments X, n_clusters and a
            random state and return an initialization.


    Available functions
    --------------------
        1. distance(self, x1, x2, metric = 'Euclid'):
            Calculate the distance between x1 and x2

            Parameters
            ----------
            1. x1 : ndarray
                The first point
            2. x2 : ndarray
                The second point
            3. metric : str, default: 'Euclid' 
                The type of distance

        2. fit(self, X)
            Fit the Kmeans model according to the given training data.
    """

    def __init__(self, k, init='k-means++'):
        self.k = k
        self.init = init

    def distance(self, x1, x2, metric='Euclid'):
        if metric == 'Euclid':
            return np.linalg.norm(x1 - x2)

    def fit(self, X):
        """
        fit(self, X)
        -----
            Fit the Kmeans model according to the given training data.
            Parameters:
            ----------
            
            1. X: ndarray, default = None
                The dataset which you want to cluster.
            
        """
        # 样本点个数
        numPoint = X.shape[0]
        # 若self.init == 'k-means++',则依次使用轮盘赌的方式选择初始聚类中心
        if self.init == 'k-means++':
            # 声明初始化类中心的列表
            center_index = []
            # new_center表示每一个新的中心(第一个类中心是随机的)
            new_center_index = np.random.randint(0, numPoint, 1)[0]
            # 将第一个new_center_index加入至center_index
            center_index.append(new_center_index)
            # 索引出new_center
            new_center = X[new_center_index]
            # 依次按照轮盘赌的方式选取剩余的中心
            # distance_storage用于存储各个样本点到已确定的类中心的距离和
            distance_storage = np.zeros(numPoint)
            # j确定确定中心点的个数
            for j in range(1, self.k):
                # i固定样本点
                for i in range(numPoint):
                    # 计算各个样本到确定的新中心的距离的最近距离
                    # 若是第一轮计算(因为初始化距离为0),或者当前计算的距离小于当前存储距离,则更新
                    if j == 1 or self.distance(X[i], new_center) < distance_storage[i]:
                        distance_storage[i] = self.distance(X[i], new_center)
                # 通过distance_storage进行轮盘赌
                # 计算概率
                p = distance_storage / distance_storage.sum()
                # 轮盘赌, replace = False为非放回采样(其实这个参数无所谓)
                new_center_index = np.random.choice(list(range(numPoint)), replace=False, p=p)
                # 将new_center_index加入至center_index中
                center_index.append(new_center_index)
                # 通过new_center_index索引new_center
                new_center = X[new_center_index]
            center_index = np.asarray(center_index)
            center = X[center_index]

        # 若self.init == 'random',随机选取中心
        elif self.init == 'random':
            # 随机中心点的Index
            center_index = np.random.randint(0, numPoint, self.k)
            # 通过中心点的index索引中心点
            center = X[center_index]
        # 初始化样本的类别
        # target 用于存储聚类的类别
        target = np.ones(numPoint)
        # pre_center用于存储上一轮的中心点
        pre_center = np.ones(center.shape, dtype=int)
        # 当一轮的中心点不变时,跳出循环
        while not np.all(center == pre_center):
            # 储存当前的中心点
            pre_center = center.copy()
            # 利用当前中心点重新聚类
            for i in range(numPoint):
                single_sample_distance = np.ones(self.k)
                for j in range(self.k):
                    single_sample_distance[j] = self.distance(X[i], center[j])
                target[i] = np.argmin(single_sample_distance)
            # 计算新的中心点
            for i in range(self.k):
                center[i] = X[target == i].mean(axis=0)
        return target


# %%
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, y = make_blobs(n_samples=1000,
                      n_features=2,
                      centers=4,
                      random_state=42)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, alpha=.6)
    # build the model
    model = KMeans(k=4)
    target = model.fit(X)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=target, s=10, alpha=.6)

# %%
