from __future__ import absolute_import

from numpy import ones_like

from LossFunc import LossFunc


class LeastSquareLoss(LossFunc):

    def g(y, y_hat):
        return y_hat - y

    def h(y, y_hat):
        return ones_like(y)
