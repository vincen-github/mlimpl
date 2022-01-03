from __future__ import absolute_import

try:
    from .XGBModel import XGBModel
    from .LossFunc import LossFunc
    from .BaseTree import BaseTree
    from .LeastSquareLoss import LeastSquareLoss
except ImportError:
    pass

__all__ = ["XGBModel",
           "LossFunc",
           "LeastSquareLoss",
           "BaseTree"
           ]
