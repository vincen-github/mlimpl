from re import findall

import gym
from matplotlib.pyplot import plot, show, figure, title, xlabel, ylabel
from numpy import zeros, argmax, arange, power
from numpy.random import choice


class QLearning:
    """
    Q-Learning is also a reinforcement learning algorithm based on temporal difference update.
    The biggest difference between it and Sarsa is that the update target of Q-learning is the optimal action
    value function Q*.
    """

    def __init__(self, env, alpha=0.1, gamma=0.9, eps=0.1):
        """
        params:
            S: State Space
            A: Action Space
            alpha: learning rate
            gamma: discount factor
            eps: parameter of ε-greedy
        """
        self.env = env

        # get state space and action space whilst record their dimension.
        self.nS = int(findall(r"\d+\.?\d*", str(self.env.observation_space))[0])
        self.S = [i for i in range(self.nS)]
        self.nA = int(findall(r"\d+\.?\d*", str(self.env.action_space))[0])
        """ - 0: move up
        - 1: move right
        - 2: move down
        - 3: move left"""
        self.A = [i for i in range(self.nA)]

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        # initialize action value function
        self.Q_star = zeros((self.nS, self.nA))

        # initialize greedy optimal policy
        self.pi = zeros((self.nS, self.nA)) + 1 / self.nA * self.eps
        for s in self.S:
            argmax_action = argmax(self.Q_star[s])
            self.pi[s, argmax_action] = self.eps / self.nA + 1 - self.eps

    def take_action(self, s):
        # Given state s, select an action.
        return choice(arange(self.nA), p=self.pi[s])

    def temporal_difference(self, s):
        # Running temporary difference once each time means playing a game.
        # time step
        t = 0
        Return = 0
        while True:
            # use current policy π select an action.
            a = self.take_action(s)
            s_prime, r, terminated, truncated, _ = self.env.step(a)
            if r == -100:
                terminated = True
            if r == 0:
                env.reset()
            Return += r
            # Q(s,a) <- Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
            self.Q_star[s, a] += self.alpha * (r + self.gamma * max(self.Q_star[s_prime]) - self.Q_star[s, a])
            s = s_prime
            t += 1
            if terminated:
                return Return

    def policy_improve(self):
        for s in self.S:
            self.pi[s] = self.eps / self.nA
            # get greedy policy
            greedy_action = argmax(self.Q_star[s])
            # update policy
            self.pi[s, greedy_action] = self.eps / self.nA + 1 - self.eps

    def run(self, s0, num_episodes):
        Returns = []
        for i in range(num_episodes):
            # use temporal_difference to evaluate current policy.
            Returns.append(self.temporal_difference(s0))
            # policy_improve
            self.policy_improve()
        return Returns


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0', render_mode='rgb_array')
    env.reset()

    # notice that env.P got from gym default set the reward in terminated state as -1 which is not distinguish the other
    # action to program our code, we need to reset the reward got in terminated state to be a number which is greater
    # than -1.
    for key1, _ in env.P.items():
        for key2, value in env.P[key1].items():
            (p, new_state, reward, terminated) = value[0]
            if terminated:
                env.P[key1][key2] = [(p, new_state, 0, terminated)]

    # parameters of instance
    eps = 0.1
    alpha = 0.1
    gamma = 0.9
    # the number of trajectory.
    num_episodes = 500

    s0 = env.start_state_index
    # print(env.start_state_index)

    agent = QLearning(env)
    Returns = agent.run(s0, num_episodes)

    figure(dpi=400)
    print("maximal return:", max(Returns))
    plot(Returns, c="darkblue")
    xlabel("num_episodes")
    ylabel("Return")
    title("Returns")
    show()

    env.close()

    # View the effectiveness of the final policy.
    MAX_GAME_NUM = 1000
    # rebuild a new environment as i can't directly reset render_mode.
    env = gym.make('CliffWalking-v0', render_mode='human')
    env.reset()
    observation = env.start_state_index
    for i in range(MAX_GAME_NUM):
        env.render()
        observation, reward, terminated, truncated, info = env.step(agent.take_action(observation))
        print("observation = {}, reward = {}, terminated = {}, truncated = {}".format(observation, reward, terminated,
                                                                                      truncated))
        if terminated or truncated:
            break
    env.close()
