from numpy import asarray, diag, sum, vstack

from adjacency_matrix import k_neighbors
from numpy.linalg import eig
from KMeans import KMeans


# from sklearn.cluster import KMeans


class SpectralClustering:
    """
    SpectralClustering:
        SpectralClustering is a class to solve graph cutting.
    ================

    @author: vincen

    @NorthWest University——CHINA Mathematics faculty statistic

    @MyGithub: https://github.com/vincen-github/mlimpl

    @MyEmail: vincen.nwu@gmail.com

    @Reference:https://www.cnblogs.com/pinard/p/6221564.html

    ------------------------

    Attribute:
    ==========
        1. block_num : int
            the number of block that u want to cut.
            block_num must be less than the number of samples.
        2. weight : func
            The measurement method between two samples.
            Common methods contain
                1. eps_neighbor:
                    A func which set eps as its hard threshold.
                    u need to pass built-in func named eps_neighbor if u want to use this func.
                    Don't forget to set the parameter eps when declare class object.
                2. k_neighbor:
                    Only set the weight of the closest k samples to non-zero.
                    u need to set the hyper—parameter k and sigma.
                3. fully_connect:
                    all weight in graph is non-zero.
                    Don't forget to set sigma when u want to use this weight.
            U can also customize the weight and pass it by this interface.
    =================
    Example:
    =========================
    """

    def __init__(self, block_num, weight):
        if isinstance(block_num, int):
            if block_num > 0:
                self.block_num = block_num
            else:
                raise ValueError("The value of block_num must be positive.")
        else:
            raise TypeError

        if hasattr(weight, '__call__'):
            self.weight = weight
        else:
            raise TypeError("weight isn't a func.")

    def fit(self, X):
        X = asarray(X)
        # get the number of samples
        n = X.shape[0]
        # check whether block_num < number of samples
        if self.block_num > n:
            raise ValueError("block_num({}) must be less than number of samples{}".format(self.block_num, n))

        # using X to build weight matrix and degree matrix
        # they are both symmetrical.So some trick for storage can be used in here
        # When building arr, it needs to be stored vertically according to the original matrix
        # arr = [self.weight(X[i], X[j]) for j in range(n) for i in range(j + 1)]

        W = k_neighbors(X, k=10, conjunction=False)
        D = sum(W.revert(), axis=1)
        # construct L = D - W
        L = diag(D) - W.revert()
        # print((L.revert() == diag(D) - W.revert()).all())
        # sqrt_D = diag(1.0 / (D ** 0.5))
        # L = sqrt_D @ L.revert() @ sqrt_D

        eig_val, eig_vec = eig(L)
        # You need to note that eig vec is a matrix and each column is an eigenvector.
        eig_val = zip(eig_val, range(len(eig_val)))
        x = sorted(eig_val, key=lambda x: x[0])

        H = vstack([eig_vec[:, i] for (v, i) in x[:self.block_num]]).T

        # kmeans = KMeans(k=self.block_num)
        # res = kmeans.fit(H)
        kmeans = KMeans(k=self.block_num)
        res = kmeans.fit(H)

        return res
