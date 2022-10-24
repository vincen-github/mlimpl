"""
Markov Reward process:
    This is the simple version of markov decision process which without make action according to π(a|s) at each step.
    agent only needs to change its state and get a deterministic reward expectation E(r|s) at each space according to
    the given state transition matrix(declared as state_trans_matrix in following code).

    Reference: https://hrl.boyuai.com/chapter/1/%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B
"""
from numpy import asarray, eye
from numpy.linalg import inv

state_trans_mat = asarray([
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])

gamma = 0.5

# rewards is a distribution under a given state at most time.
# In this case, we formulize it as a deterministic function w.r.t state.
reward_expectations = lambda state: [-1, -2, -2, 10, 1, 0][state]


def calc_return(state_chain, gamma):
    """calculate expectation of return given a state chain. """
    Return = 0
    for state in reversed(state_chain):
        Return = gamma * Return + reward_expectations(state)

    return Return


# calculate analytic value function of MRP
def analytic_val_func(state_trans, reward_expectation, gamma):
    state_num = state_trans.shape[0]
    """solve analytic solution of Value function through Bellman equation."""
    Reward_expectations = asarray([reward_expectations(state) for state in range(state_num)]).reshape(-1, 1)
    val_func = lambda state: (inv(eye(state_num) - gamma * state_trans) @ Reward_expectations)[state, 0]
    return val_func


if __name__ == "__main__":
    state_chain = [0, 1, 2, 5]
    print(calc_return(state_chain, gamma))

    val_func = analytic_val_func(state_trans_mat, reward_expectations, gamma)

    print([val_func(state) for state in range(6)])
