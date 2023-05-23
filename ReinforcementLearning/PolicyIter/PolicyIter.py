from re import findall

import gym
from numpy import ones, zeros, argmax


class ShapeError(Exception):
    def __init__(self):
        super(Exception, self).__init__()


class PolicyIter(object):
    """This class is the implementation of policy iteration to solve model based reinforcement learning problem, thus u
    need to pass the transition matrix of mdp, of the form which same as gym's. This program will rely on following
    process to find a great policy what ur need.
    policy evaluation -> policy improve -> policy evaluation -> policy improve -> .... (until convergence)
    """

    def __init__(self, S, A, P, gamma, threshold, init_pi=None):
        """
        params:
            1. S: state space
            2. A: action space
            3. P: the transition matrix and Rewards whose form is same as CliffWalkingEnv implementation in gym.
            4. gamma: discount factor
            5. threshold: the stop condition for the iteration of Value function
            5. init_pi: initial policy, this para will set average policy if u don't pass it into class.

        """
        self.S = S
        self.A = A

        self.nS = len(self.S)
        self.nA = len(self.A)

        self._P = P
        self.__get_trans_prob_and_r_sa()

        # print(self.trans_prob(0, 0, 0), self.trans_prob(0, 1, 1), self.trans_prob(0, 2, 0), self.trans_prob(0, 2, 12))

        self.gamma = gamma
        self.threshold = threshold

        self.pi = init_pi

        # Value function
        self.V = zeros(nS)

    @property
    def P(self):
        return self._P

    @P.setter
    def P(self, P):
        if not isinstance(P, dict):
            raise TypeError
        if not len(P) == self.nS:
            raise ShapeError
        else:
            self._P = P

    @property
    def pi(self):
        return self._pi

    @pi.setter
    def pi(self, pi):
        if pi is None:
            self._pi = dict()
            # if the initial policy passed is not None, automatically indicate policy is uniform.
            for s in self.S:
                self._pi[s] = ones(self.nA) / self.nA
        elif not isinstance(pi, dict):
            raise TypeError
        else:
            self._pi = pi

    def __get_trans_prob_and_r_sa(self):
        """The method to calculate trans_prob(s, a) and r(s, a) so that we can update value function."""
        # transition probability rely on (s, a, s')
        self.trans_tensor = zeros((self.nS, self.nA, self.nS))
        # reward is depend on (s, a, s')
        self.rewards_tensor = zeros((self.nS, self.nA, self.nS))
        # record the expectation of R given (s, a)
        self.r_sa_mat = zeros((self.nS, self.nA))

        for s in self.S:
            # self.P[s][a] = [(p(1), s'(1), R(1), done(1)), (p(2), s'(2), R(2), done(2)), ...]
            for a in self.A:
                for p, s_prime, R, done in self.P[s][a]:
                    self.trans_tensor[s][a][s_prime] = p
                    self.rewards_tensor[s][a][s_prime] = R
                    self.r_sa_mat[s][a] += p * R

        # record the expectation of reward given s and a
        self.trans_prob = lambda s, a, s_prime: self.trans_tensor[s][a][s_prime]
        self.R = lambda s, a, s_prime: self.rewards_tensor[s][a][s_prime]
        self.r = lambda s, a: self.r_sa_mat[s][a]

    def evaluate(self):
        while True:
            # initial difference as 0
            diff = 0
            for s in self.S:
                old_V = self.V.copy()
                # update V[s]
                self.V[s] = 0
                for a in self.A:
                    # Note: In fact, Here using Gauss–Seidel iterative method to accelerate convergence.
                    # Bellman Expectation Equation
                    # Vπ(s) = Σ_{a∈A}π(a|s)(r(s,a) + γΣ_{s'∈S}p(s'|s,a)Vπ(s'))
                    self.V[s] += self.pi[s][a] * (self.r(s, a) + \
                                                  self.gamma * sum(
                                [self.trans_prob(s, a, s_prime) * old_V[s_prime] for s_prime in S])
                                                  )
                diff = max(diff, abs(old_V[s] - self.V[s]))
            # if || V - V_old ||∞ < threshold, it implies that we have found the fixed point of bellman equation
            # corresponding to current policy π,jump out from loop.
            if diff < self.threshold:
                break

    def improve(self):
        for s in self.S:
            # π(s) = argmax(Qπ(s,a))
            # Qπ(s,a) = r(s,a) + γP(s'|s,a)Vπ(s')
            max_index = argmax(
                [self.r(s, a) + self.gamma * sum([self.trans_prob(s, a, s_prime) * self.V[s_prime] for s_prime in S])
                 for a in self.A])
            # new_π = (0, 0, ..., 0, 1, 0, ..., 0), where the index of nonzero item is argmax(Qπ(s,a)).
            self.pi[s] = zeros(self.nA)
            self.pi[s][max_index] += 1

    def iter(self) -> tuple:
        while True:
            old_pi = self.pi.copy()
            # evaluate current policy
            self.evaluate()
            self.improve()
            # compare new policy and old policy, as the values in them is all list, we can't direct compare them, so we
            # use a loop to implement it.
            if all([(self.pi[s] == old_pi[s]).all() for s in self.S]):
                return self.pi, self.V


if __name__ == '__main__':
    env = gym.make('CliffWalking-v0', render_mode='human')
    env.reset()

    # get params needed to pass to class.
    P = env.P
    print(P[0])
    # notice that P got from gym default set the reward in terminated state as -1 which is not distinguish the other
    # action to program our code, we need to reset the reward got in terminated state to be a number which is greater
    # than -1.
    for key1, _ in P.items():
        for key2, value in P[key1].items():
            (p, new_state, reward, terminated) = value[0]
            if terminated:
                P[key1][key2] = [(p, new_state, 0, terminated)]
    print(P)

    nS = int(findall(r"\d+\.?\d*", str(env.observation_space))[0])
    S = [i for i in range(nS)]
    nA = int(findall(r"\d+\.?\d*", str(env.action_space))[0])
    """ - 0: move up
    - 1: move right
    - 2: move down
    - 3: move left"""
    A = [i for i in range(nA)]
    GAMMA = 0.9
    THRESHOLD = 1e-3
    MAX_GAME_NUM = 1000

    # creat a deterministic policy
    # pi = dict()
    # for s in S:
    #     pi[s] = [0, 1, 0, 0]

    obj = PolicyIter(S, A, P, GAMMA, THRESHOLD)
    pi_star, V_star = obj.iter()

    print(pi_star)
    print(V_star)

    observation = env.start_state_index
    for i in range(MAX_GAME_NUM):
        env.render()

        action = argmax(pi_star[observation])
        observation, reward, terminated, truncated, info = env.step(action)
        print("observation = {}, reward = {}, terminated = {}, truncated = {}".format(observation, reward, terminated,
                                                                                      truncated))
        if terminated or truncated:
            break
    env.close()
