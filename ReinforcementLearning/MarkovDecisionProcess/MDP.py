"""
    Markov Decision Process:
        MDP is the more complex version of MRP, which consider how to choose the action at each step.
    Reference: https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B
"""

# state number
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

    # To calculate the state value function of mdp, we transform it as solving Bellman equation problem of mrp
    def __trans_mdp_to_mrp(self):
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
        # firstly convert R_s and P_S to matrix form.
        P_S_mat = convert_from_dict_to_matrix(self.P_S)

        R_S_mat = convert_from_dict_to_matrix(self.R_S)

        """solve analytic solution of Value function through Bellman equation."""
        self.val_func = lambda s: (inv(eye(P_S_mat.shape[0]) - self.gamma * P_S_mat) @ R_S_mat)[int(s.split("s")[1]) - 1, 0]
        return self.val_func


gamma = 0.5
if __name__ == "__main__":
    gamma = 0.5

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