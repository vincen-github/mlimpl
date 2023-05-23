from spectral_clustering import *
from adjacency_matrix import k_neighbors
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# make some data
seed(1)
X, y = make_circles(500, factor=0.5, noise=0.05)

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], s=10, c=y)
# plt.show()

sc = SpectralClustering(block_num=2, weight=k_neighbors)
#
res = sc.fit(X)
#
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', c=res, cmap='brg')
plt.show()
