"""
    Markov Decision Process:
        MDP is the more complex version of MRP, which consider how to choose the action at each step.
    Reference: https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B
"""

# state number
from random import choice, uniform

from numpy import zeros, asarray, eye
from numpy.linalg import inv


def convert_from_dict_to_matrix(dic):
    # If the key of input dict is a two-dimensional tuple, construct a matrix;
    # else if it is a one-dimensional tuple, construct a vector.
    # analysis the shape of matrix
    max_index = 0
    mat_elems = []
    if isinstance(list(dic.keys())[0], tuple):
        for (key_1, key_2), val in dic.items():
            row_index = int(key_1.split("s")[1])
            col_index = int(key_2.split("s")[1])
            mat_elems.append((row_index - 1, col_index - 1, val))
            max_index = max(row_index, col_index, max_index)
        mat = zeros((max_index, max_index))
        for elem in mat_elems:
            mat[elem[:2]] = elem[-1]
        return asarray(mat)

    if isinstance(list(dic.keys())[0], str):
        for key, val in dic.items():
            index = int(key.split("s")[1])
            mat_elems.append((index - 1, val))
            max_index = max(index, max_index)
        mat = zeros((max_index, 1))
        for elem in mat_elems:
            mat[elem[0], 0] = elem[1]
        return asarray(mat)


class MDP:
    def __init__(self, S, A, P_S_A, Rewards_S_A, Pi, gamma):
        self.S = S
        self.A = A
        # It is agreed to use capital letters such as P and R to indicate the transition probability
        # and reward of data structure in the form of dictionary or matrix. Conversely,  Use lowercase letters to
        # represent which of the function form.
        # p(s'|s, a) = P_S_A[(s, a, s')]
        self.P_S_A = P_S_A
        # reward function, R_S_A[(s, a)] means that the reward you will get if you taking action a under state s.
        # Please distinguish the difference between it and the following single step expected reward r_S.
        self.Rewards_S_A = Rewards_S_A
        # π(a|s)
        self.Pi = Pi

        self.gamma = gamma

        self.state_num = len(S)
        self.action_num = len(A)

        self.__build_R_S_A()

        self.__trans_mdp_to_mrp()

    # get expected reward in single step r(s, a) = E{R_S_A[s, a]|s, a}
    def __build_R_S_A(self):
        self.R_S_A = dict([])
        for key, val in self.P_S_A.items():
            s, a, s_prime = key
            if (s, a) in self.R_S_A.keys():
                self.R_S_A[(s, a)] += val * self.Rewards_S_A[(s, a)]
            else:
                self.R_S_A[(s, a)] = val * self.Rewards_S_A[(s, a)]

    def __trans_mdp_to_mrp(self):
        """
        Model Based!!
        To calculate the state value function of mdp, we transform it as solving Bellman equation problem of mrp.
        """
        self.R_S, self.P_S = dict([]), dict([])

        # calculate P_S
        for key, val in self.P_S_A.items():
            # key = (s, a, s')
            s, a, s_prime = key[0], key[1], key[2]
            pi_a_s = self.Pi[(a, s)]
            # p(s'|s) = Σπ(a|s)P(s'|s,a)
            if (s, s_prime) in self.P_S.keys():
                self.P_S[(s, s_prime)] += pi_a_s * self.P_S_A[(s, a, s_prime)]
            else:
                self.P_S[(s, s_prime)] = pi_a_s * self.P_S_A[(s, a, s_prime)]
        # Calculate R_s
        # !!!# Note that you can't directly splice this and  above loop into a single w.r.t self. P_S_A.items().
        # because self.P_S_A has three elements corresponding to the key ("s4", "-> with prob").
        # R_S["s4"] will computed repetitively.
        for key, val in self.R_S_A.items():
            s, a = key[0], key[1]
            pi_a_s = self.Pi[(a, s)]
            # r(s) = Σπ(a|s)r(s,a)
            if s in self.R_S.keys():
                self.R_S[s] += pi_a_s * val
            else:
                self.R_S[s] = pi_a_s * val

    def analytic_value_func(self):
        """Model Based!!!"""
        # firstly convert R_s and P_S to matrix form.
        P_S_mat = convert_from_dict_to_matrix(self.P_S)

        R_S_mat = convert_from_dict_to_matrix(self.R_S)

        """solve analytic solution of Value function through Bellman equation."""
        self.val_func = lambda s: (inv(eye(P_S_mat.shape[0]) - self.gamma * P_S_mat) @ R_S_mat)[
            int(s.split("s")[1]) - 1, 0]
        return self.val_func

    def sample(self, max_step: int):
        episode = []
        # get initial state
        current_s = choice(self.S)
        for step in range(max_step):
            # get available action and its prob about current s
            available_a_p = [(a, p) for (a, s), p in self.Pi.items() if s == current_s]
            # select action under current state through π(a|s)
            acc_prob, current_a = 0, None
            for a, p in available_a_p:
                acc_prob += p
                if uniform(0, 1) < acc_prob:
                    current_a = a
                    break
            # get reward
            current_r = None
            for (s, a), r in self.R_S_A.items():
                if s == current_s and a == current_a:
                    current_r = r
                    break
            # add (current action, current state, current reward) into episode
            episode.append((current_s, current_a, current_r))
            # update new state through current action and state
            # get reachable state under current action adn state
            reachable_s_p = [(s_prime, p) for (s, a, s_prime), p in self.P_S_A.items() if
                             s == current_s and a == current_a]
            acc_prob = 0
            for s_prime, p in reachable_s_p:
                acc_prob += p
                if uniform(0, 1) < acc_prob:
                    current_s = s_prime
                    break
        return episode

    def monte_carlo(self, episodes):
        """
            Model Free Reinforcement Learning!!!!
            using monte carlo method to estimate the value of Vπ(s). This method use offline data to estimate Vπ(s).
            The initial form of this algorithm .
            这个算法的初始形式应为在episodes中遍历每一条轨道,当第i条轨道的起始状态是s时,s状态的计数器加1: N[s] = N[s] + 1.
            利用该轨道的reward去计算return: G[s][i] = Rt + γR(t + 1) + γ²R(t+2) + ...  (i指代第i条轨道, s指代给定的状态)
            我们现实中不可能得到无穷序列的离线数据， 但折扣因子γ < 1的设置使得我们可以将上级数截断,使用部分和替代该无穷级数.
            遍历完所有episodes中的轨道后,利用 Vπ[s] = Σ_{i} G[s][i] / N[s]去估计Vπ在s处的取值.
            但很显然，这种方式对于episodes的数量要求很苛刻.

            一个改进点即是一但我们遇到状态s,不论其是否为起始状态,我们都将s后面的轨道部分其视为以状态s为起始的一条链.(为了在这种setup下我们可以即时更新，不得不从后往前去计算回报。)

            第二个改进点是增量更新:
                1/(m+1) Σ_{i=1}^{m+1} Xi = 1/(m+1) X_{m+1} + (1 - 1/(m+1)) Σ_{i=1}^{m} Xi


        """
        N = dict([(s, 0) for s in self.S])
        V = dict([(s, 0) for s in self.S])
        for episode in episodes:
            G = 0
            for i in range(len(episode) - 1, -1, -1):
                # We must calculate returns from tail to the front because the first improvement mentioned as above.
                # If reverse the order of calculation, your can't immediately compute the return when your encounter a
                # state that your can't incremental update.
                (s, a, r) = episode[i]
                G = r + self.gamma * G
                N[s] += 1
                # Incremental update
                V[s] = V[s] + (G - V[s]) / N[s]
        return V

    # calculate occupancy given s and a
    def occupancy(self, s, a, episodes):
        rho = 0
        # ρ(s, a) = (1 - γ)Σ(γ^t)Pt(s, a) ≈ (1 - γ)Σ(γ^t)Nt(s, a)/Nt
        # First get max_step Nt in episodes
        max_step = max([len(episode) for episode in episodes])
        # Declare total_times to record Nt in each t.
        # This is necessary as the length of episode is not equal.
        total_times = zeros(max_step)
        # declare occur_times to record Nt(s, a)
        occur_times = zeros(max_step)
        for episode in episodes:
            for i in range(len(episode)):
                s_prime, a_prime, r = episode[i]
                total_times[i] += 1
                if s == s_prime and a == a_prime:
                    occur_times[i] += 1
        for t in range(max_step):
            rho += self.gamma ** t * occur_times[t] / total_times[t]
        return self.gamma * rho


if __name__ == "__main__":
    gamma = 0.1

    state_num = 5

    # state_space
    state_space = ["s" + str(i + 1) for i in range(state_num)]

    # action space
    action_space = ["-> s1", "-> s2", "-> s3", "—> s4", "-> s5", "-> with prob"]

    # compress a 3-dim state transition matrix and reward tensor to 1-dim.
    # each element of env of form: { (s, a, s') : P(s'|s, a)}
    # explain: if you are now in state "s1" and select action "-> with prob" by a given policy, the state will transform
    # to state "s2" from "s1" with probability 0.2, to "s3" with probability 0.4 and to "s4" with probability with 0.4.
    P_S_A = {
        ("s1", "-> s1", "s1"): 1.0,
        ("s1", "-> s2", "s2"): 1.0,
        ("s2", "-> s1", "s1"): 1.0,
        ("s2", "-> s3", "s3"): 1.0,
        ("s3", "-> s4", "s4"): 1.0,
        ("s3", "-> s5", "s5"): 1.0,
        ("s4", "-> s5", "s5"): 1.0,
        ("s5", "-> s5", "s5"): 1.0,
        ("s4", "-> with prob", "s2"): 0.2,
        ("s4", "-> with prob", "s3"): 0.4,
        ("s4", "-> with prob", "s4"): 0.4,
    }

    Rewards_S_A = {
        ("s1", "-> s1"): -1,
        ("s1", "-> s2"): 0,
        ("s2", "-> s1"): -1,
        ("s2", "-> s3"): -2,
        ("s3", "-> s4"): -2,
        ("s3", "-> s5"): 0,
        ("s4", "-> s5"): 10,
        ("s4", "-> with prob"): 1,
        ("s5", "-> s5"): 0,
    }

    Pi = {
        ("-> s1", "s1"): 0.5,
        ("-> s2", "s1"): 0.5,
        ("-> s1", "s2"): 0.5,
        ("-> s3", "s2"): 0.5,
        ("-> s4", "s3"): 0.5,
        ("-> s5", "s3"): 0.5,
        ("-> s5", "s4"): 0.5,
        ("-> with prob", "s4"): 0.5,
        ("-> s5", "s5"): 1
    }

    mdp = MDP(state_space, action_space, P_S_A, Rewards_S_A, Pi, gamma)
    print(mdp.P_S)
    print(mdp.R_S)

    # mat = convert_from_dict_to_matrix(mdp.P_S)
    # print(mat)
    #
    # vec = convert_from_dict_to_matrix(mdp.R_S)
    # print(vec)
    val_func = mdp.analytic_value_func()
    for s in state_space:
        print(val_func(s))

    episodes = [mdp.sample(max_step=10) for i in range(2000)]
    V_pi = mdp.monte_carlo(episodes)

    print(V_pi)

    print(mdp.occupancy("s4", "-> with prob", episodes))
