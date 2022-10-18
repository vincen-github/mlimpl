from abc import ABCMeta, abstractmethod


class EnvInterface(metaclass=ABCMeta):

    @abstractmethod
    def get_reward(self):
        pass
