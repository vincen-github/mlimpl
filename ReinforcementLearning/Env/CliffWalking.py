from abc import ABC

from ReinforcementLearning.Env.EnvInterface import EnvInterface


class CliffWalking(EnvInterface, ABC):
    """
        CliffWalking is a simple environment of reinforcement learning.
    """

    def __init__(self):
        pass

    def take_reward(self):
        pass
