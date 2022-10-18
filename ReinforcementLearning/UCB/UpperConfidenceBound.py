from copy import copy

from matplotlib.pyplot import plot, show
from numpy import zeros, argmax, cumsum, asarray, sqrt, log


class UCB:
    """
    Reference: https://hrl.boyuai.com/chapter/1/%E5%A4%9A%E8%87%82%E8%80%81%E8%99%8E%E6%9C%BA#24-%CF%B5-%E8%B4%AA%E5%BF%83%E7%AE%97%E6%B3%95
    """

    def __init__(self, k, env, c=15, if_log=False):
        """
        Note: The parameter c should be set as the same magnitude as expected reward,
        otherwise the same action will always occur.
        """
        # k is the size of action space
        self.k = k
        # weight of upper bound
        self.c = c
        # Estimated conditional expectation of R given A
        self.Q = zeros(k)
        # counter for A = a
        self.N = zeros(k)
        # upper bound
        self.U = zeros(k)
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
        action = argmax(self.Q + self.c * self.U)
        reward = self.env.get_reward(action)

        return action, reward

    def run(self, max_step=1000):

        action_ls, reward_ls, Q_ls, U_ls = [], [], [], []
        for step in range(1, max_step + 1):
            action, reward = self.take_step()

            self.N[action] += 1
            self.Q[action] += (reward - self.Q[action]) / self.N[action]

            # update U
            self.U = sqrt(log(step) / (2 * self.N + 1))

            if self.if_log:
                action_ls.append(action)
                reward_ls.append(reward)
                Q_ls.append(copy(self.Q))
                U_ls.append(copy(self.U))

        if self.if_log:
            self.log.update([("action", action_ls), ("reward", reward_ls), ("ucb", U_ls), ("Q", Q_ls)])
            return self.Q, self.U, self.log
        else:
            return self.Q, self.U


if __name__ == "__main__":
    from ReinforcementLearning.Env.MultiArmedBandit import MultiArmedBandit

    mba = MultiArmedBandit()
    # print(mba.take_reward(2))

    MAX_STEP = 2000

    eg = UCB(k=3, env=mba, if_log=True)
    est_Q, est_U, log = eg.run(max_step=MAX_STEP)

    print(log["action"])
    print(log["ucb"])


    def plot_Q(log):
        Q_ls = log["Q"]
        plot(Q_ls)
        show()


    plot_Q(log)


    def plot_U(log):
        U_ls = log["ucb"]
        plot(U_ls)
        show()


    plot_U(log)


    # calculate cumulative regret
    def plot_cumulative_regret(best_expect_reward, log):
        regrets = best_expect_reward - asarray(log["reward"])

        plot(cumsum(regrets))
        show()


    plot_cumulative_regret(best_expect_reward=max(mba.reward_expectation), log=log)
