from matplotlib.pyplot import plot, show
from numpy import ones, exp
from numpy.random import choice


def softmax(x):
    return exp(x) / sum(exp(x))


class PolicyGradient:
    """
        PolicyGradient is another method whose idea is different from Value Function Method.
        It optimizes the action distribution to maximize E_{A~π}[E(R|A)].
        we use softmax function to simplify this constrained optimization problem.

        Reference: 2022 Summer Short Course in TianYuan Mathematical Center in Central China:
            Mathematical Theory and Applications of Deep Learning
            Prof. Haizhao Yang (University of Maryland,CollegePark)
            Course Video :Course(3) replay in http://tmcc.whu.edu.cn/info/1262/2052.htm
        """

    def __init__(self, k, alpha):
        self.k = k
        self.H = ones(self.k)
        self.pi = ones(self.k) / self.k
        self.alpha = alpha

        # accumulative reward
        self.acc_reward = 0

    def update(self, env, max_step):
        average_reward = []
        for step in range(max_step):
            action = choice(range(self.k), p=self.pi)
            reward = env.take_reward(action)

            # update
            self.H -= self.alpha * (reward - self.acc_reward / (step + 1)) * self.pi

            self.H[action] += self.alpha * (reward - self.acc_reward / (step + 1))

            self.pi = softmax(self.H)

            self.acc_reward += reward

            average_reward.append(self.acc_reward / (step + 1))

        return average_reward


if __name__ == "__main__":
    from ReinforcementLearning.Env.MultiArmedBandit import MultiArmedBandit

    mab = MultiArmedBandit()
    pg = PolicyGradient(k=3, alpha=0.01)

    max_step = 10000

    average_reward = pg.update(mab, max_step)

    print(pg.pi)
    plot(list(range(max_step)), average_reward)
    show()
