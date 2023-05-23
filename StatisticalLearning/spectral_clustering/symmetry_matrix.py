# Declare a symmetric matrix class to save storage space
from numpy import sqrt, zeros, diag


class SymmetryMatrix:
    def __init__(self, arr):
        self.arr = arr

    def __getitem__(self, item):
        index = int(max(item) * (max(item) + 1) / 2 + min(item))
        return self.arr[index]

    def __setitem__(self, key, value):
        index = int(max(key) * (max(key) + 1) / 2 + min(key))
        self.arr[index] = value
        return self.arr

    def revert(self):
        # revert a matrix with shape n×n
        # calculate n
        n = int((sqrt(1 + 8 * len(self.arr)) - 1) / 2)
        mat = zeros(shape=(n, n))
        for j in range(n):
            for i in range(j + 1):
                mat[i, j] = self[i, j]
        # Copy the upper triangle area to the lower triangle area
        mat += mat.T - diag(mat.diagonal())
        return mat
