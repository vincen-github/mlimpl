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
            self.reward_expectation = [10, 20, 30]
        if not reward_variance:
            self.reward_variance = [1, 1, 1]

    def take_reward(self, action):
        return normal(self.reward_expectation[action], self.reward_variance[action])
