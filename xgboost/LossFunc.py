from abc import ABC, abstractmethod


class LossFunc(ABC):
    """
    Interface that implement by the loss function passed into class XGModel.
    ===================================

    Method implemented:
        1. g(self, y, y_hat) : function, abstract and static
            The first derivative function of the loss function
        2. h(self, y, y_hat) : function, abstract and static
            The second derivative function of the loss function
    ----------------------------------
    """

    @staticmethod
    @abstractmethod
    def g(y, y_hat):
        pass

    @staticmethod
    @abstractmethod
    def h(y, y_hat):
        pass
