from numpy import exp, zeros, sum, asarray, sqrt
from heapq import nsmallest
from symmetry_matrix import SymmetryMatrix


def k_neighbors(X, k=10, sigma=1, conjunction=True):
    """
    k_neighbors:
        using k_neighbors to build adjacent matrix.it means that only the weight of kth-nearest samples
        will be nonzero for a fixed sample xi. sigma indicates the variance of rbf function. By setting conjunction as
        True or False to decide hot to build a symmetry adjacent matrix.
        For more detail,please refer to https://www.cnblogs.com/pinard/p/6221564.html.
    """
    X = asarray(X)
    # get sample numbers
    n = X.shape[0]

    nsmallest_index = []
    for i in range(n):
        # calculate k-nearest samples,You have to note that the Euclidean distance needs to be used here
        # instead of the rbf kernel as below.Otherwise, the features captured by KNN will be different,
        # resulting in problems with the results.
        dist = asarray([sqrt(sum((X[i] - X[j]) ** 2)) for j in range(n)])
        # find index of the nearest samples for all samples(k + 1是因为除去了自己)
        nsmallest_index.append(nsmallest(k + 1, range(n), dist.take))

    # get adjacency matrix through above index set
    arr = zeros(int(n * (n + 1) / 2))
    adjacency_matrix = SymmetryMatrix(arr)
    for i in range(n):
        for j in nsmallest_index[i]:
            # if conjunction == True
            if conjunction:
                # for a pair index (i, j), W[i, j] != 0 if and only if xi in knn(xj) and xj in knn(xi)
                if i in nsmallest_index[j]:
                    adjacency_matrix[i, j] = rbf(X[i], X[j], sigma=sigma)
            else:
                adjacency_matrix[i, j] = rbf(X[i], X[j], sigma=sigma)
    return adjacency_matrix


def rbf(x, y, sigma=1):
    s = sqrt(sum((x - y) ** 2))
    return exp(-s / 2 / sigma ** 2)
