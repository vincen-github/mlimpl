from copy import copy
from random import uniform, randint

from matplotlib.pyplot import plot, show
from numpy import zeros, argmax


class EpsilonGreedy:
    """
    EpsilonGreedy is an implement class of the simplest reinforcement learning method which
    is suitable for single state and finite action(denoted as k in here).It is a revising version
    of greedy method. Its creation is used to avoid Local maximizer problem.

    Reference: 2022 Summer Short Course in TianYuan Mathematical Center in Central China:
        Mathematical Theory and Applications of Deep Learning
        Prof. Haizhao Yang (University of Maryland,CollegePark)
        Course Video :Course(3) replay in http://tmcc.whu.edu.cn/info/1262/2052.htm
    """
    def __init__(self, k, eps=0.1):
        self.k = k
        # Estimating conditional expectation of R given A
        self.Q = zeros(k)
        # counter for A = a
        self.N = zeros(k)
        # hyperparameter of EpsilonGreedy
        self.eps = eps

    def update(self, env, max_step=1000):
        Q_ls = [copy(self.Q)]

        for step in range(max_step):
            action = randint(0, self.k - 1) if uniform(0, 1) < self.eps else argmax(self.Q)
            reward = env.take_reward(action)

            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]

            Q_ls.append(copy(self.Q))

        return Q_ls


if __name__ == "__main__":
    from ReinforcementLearning.Env.MultiArmedBandit import MultiArmedBandit

    mba = MultiArmedBandit()
    # print(mba.take_reward(2))

    max_step = 2000

    eg = EpsilonGreedy(k=3)
    Q_ls = eg.update(env=mba, max_step=max_step)

    plot(list(range(max_step + 1)), Q_ls)
    show()
