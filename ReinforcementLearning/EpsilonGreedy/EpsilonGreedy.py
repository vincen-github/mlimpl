from copy import copy
from random import uniform, randint

from matplotlib.pyplot import plot, show
from numpy import zeros, argmax, cumsum, asarray


class EpsilonGreedy:
    """
    EpsilonGreedy is an implement class of the simplest reinforcement learning method which
    is suitable for single state and finite action(denoted as k in here).It is a revising version
    of greedy method. Its creation is used to avoid Local maximizer problem.

    Reference: 1. 2022 Summer Short Course in TianYuan Mathematical Center in Central China:
        Mathematical Theory and Applications of Deep Learning
        Prof. Haizhao Yang (University of Maryland,CollegePark)
        Course Video :Course(3) replay in http://tmcc.whu.edu.cn/info/1262/2052.htm
        2. https://hrl.boyuai.com/chapter/1/%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA#24-%CF%B5-%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95
    """

    def __init__(self, k, env, eps=0.1, if_log=False):
        self.k = k
        # Estimated conditional expectation of R given A
        self.Q = zeros(k)
        # counter for A = a
        self.N = zeros(k)
        # hyperparameter of EpsilonGreedy
        self.eps = eps
        # environment
        self.env = env

        self.if_log = if_log
        # log
        if self.if_log:
            self.log = dict()

    def take_step(self):
        """
        This method is used to interact with env on single step.
        """
        # Select historically optimal action with probability 1-eps and otherwise evenly select one from  Action space.
        action = randint(0, self.k - 1) if uniform(0, 1) < self.eps else argmax(self.Q)
        reward = self.env.get_reward(action)

        return action, reward

    def run(self, max_step=1000):

        action_ls, reward_ls, Q_ls = [], [], []
        for step in range(max_step):
            action, reward = self.take_step()

            self.N[action] += 1
            # [Q in step k] = [Q in step k - 1] + (reward  - [Q in step k - 1])/k
            self.Q[action] += (reward - self.Q[action]) / self.N[action]

            if self.if_log:
                action_ls.append(action)
                reward_ls.append(reward)
                Q_ls.append(copy(self.Q))

        if self.if_log:
            self.log.update([("action", action_ls), ("reward", reward_ls), ("Q", Q_ls)])
            return self.Q, self.log
        else:
            return self.Q


if __name__ == "__main__":
    from ReinforcementLearning.Env.MultiArmedBandit import MultiArmedBandit

    mba = MultiArmedBandit()
    # print(mba.take_reward(2))

    MAX_STEP = 500

    eg = EpsilonGreedy(k=3, env=mba, if_log=True)
    est_Q, log = eg.run(max_step=MAX_STEP)


    def plot_Q(log):
        Q_ls = log["Q"]
        plot(Q_ls)
        show()


    plot_Q(log)

    # calculate cumulative regret
    def plot_cumulative_regret(best_expect_reward, log):
        regrets = best_expect_reward - asarray(log["reward"])

        plot(cumsum(regrets))
        show()


    plot_cumulative_regret(best_expect_reward=max(mba.reward_expectation), log=log)
