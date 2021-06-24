from abc import ABC, abstractmethod


class Proposal_Distribution(ABC):

    @staticmethod
    @abstractmethod
    def pdf(x, y):
        pass

    @staticmethod
    @abstractmethod
    def sampling():
        pass
