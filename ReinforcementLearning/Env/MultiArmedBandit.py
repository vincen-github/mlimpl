from abc import ABC

from numpy.random import normal

from ReinforcementLearning.Env.EnvInterface import EnvInterface


class MultiArmedBandit(EnvInterface, ABC):
    """
    MultiArmedBandit simple environment of reinforcement learning.
    It only has a single State and finite Action(denoted as k).
    We assume that the reward distribution is normal distribution in this implementation.
    u can  set the expectation and variance value of them manually"""

    def __init__(self, reward_expectation=None, reward_variance=None):
        if not reward_expectation:
            self.reward_expectation = [10, 13, 15]
        else:
            self.reward_expectation = reward_expectation
        if not reward_variance:
            self.reward_variance = [8, 5, 3]
        else:
            self.reward_variance = reward_variance

    def get_reward(self, action):
        return normal(self.reward_expectation[action], self.reward_variance[action])
