from copy import copy

from matplotlib.pyplot import plot, show
from numpy import argmax, sqrt, log, where

from ReinforcementLearning.EpsilonGreedy.EpsilonGreedy import EpsilonGreedy


class UpperConfidenceBoundSelection(EpsilonGreedy):
    """
    Another reinforcement learning method liking EpsilonGreedy for avoiding Local Minimizer problem in Greedy Algorithm.
    Reference: 2022 Summer Short Course in TianYuan Mathematical Center in Central China:
        Mathematical Theory and Applications of Deep Learning
        Prof. Haizhao Yang (University of Maryland,CollegePark)
        Course Video :Course(3) replay in http://tmcc.whu.edu.cn/info/1262/2052.htm
    """

    def __init__(self, k, c):
        super(UpperConfidenceBoundSelection, self).__init__(k)
        # hyperparameter which controls the update formal of action
        self.c = c

    def update(self, env, max_step=1000):
        Q_ls = [copy(self.Q)]

        for step in range(max_step):
            # If N[i] = 0, then i is considered to be a maximizing action.
            action = argmax(self.Q + self.c * sqrt(log(step + 1) / self.N)) if all(self.N) else where(self.N == 0)[0][0]
            reward = env.take_reward(action)

            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]

            Q_ls.append(copy(self.Q))
        print(self.N)

        return Q_ls


if __name__ == "__main__":
    from ReinforcementLearning.Env.MultiArmedBandit import MultiArmedBandit

    mba = MultiArmedBandit()
    # print(mba.take_reward(2))

    max_step = 10000

    # c is important for convergence.
    ucbs = UpperConfidenceBoundSelection(k=3, c=100)
    Q_ls = ucbs.update(env=mba, max_step=max_step)

    plot(list(range(max_step + 1)), Q_ls)
    show()
