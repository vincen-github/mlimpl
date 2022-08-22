from abc import ABCMeta, abstractmethod


class EnvInterface(metaclass=ABCMeta):

    @abstractmethod
    def take_reward(self):
        pass
