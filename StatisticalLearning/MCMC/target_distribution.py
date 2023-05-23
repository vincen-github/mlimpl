from abc import ABC, abstractmethod


class Target_Distribution(ABC):

    @staticmethod
    @abstractmethod
    def pdf(x):
        pass
