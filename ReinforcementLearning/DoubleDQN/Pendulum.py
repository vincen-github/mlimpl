import random

import numpy as np
import torch
from matplotlib.pyplot import plot, show, figure, xlabel, ylabel
from numpy import mean

from ReinforcementLearning.DoubleDQN.DoubleDQN import DoubleDQN
from ReinforcementLearning.DoubleDQN.ReplayBuffer import ReplayBuffer
from ReinforcementLearning.DoubleDQN import HyperParams
import gym

def trans_discrete_to_continuous(discrete_action, env, action_card):
    # minimum value of continuous action
    action_lower_bound = env.action_space.low[0]
    # maximum value of continuous action
    action_upper_bound = env.action_space.high[0]
    return action_lower_bound + (discrete_action / (action_card - 1)) * (action_upper_bound - action_lower_bound)


if __name__ == "__main__":
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    # Environment Reference: https://www.gymlibrary.dev/environments/classic_control/pendulum/
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


    # get instance of Class ReplayBuffer
    replay_buffer = ReplayBuffer(HyperParams.buffer_capacity)

    state_dim = env.observation_space.shape[0]
    # divide continuous actions space into 11 discrete actions.
    action_card = 11

    agent = DoubleDQN(state_dim, HyperParams.hidden_dim, action_card, HyperParams.lr, HyperParams.gamma, HyperParams.eps,
                HyperParams.target_update_frequency, HyperParams.device)

    returns = []

    for i in range(HyperParams.num_episodes):
        episode_return = 0
        state = env.reset(seed=0)[0]
        while True:
            action = agent.take_action(state)
            # transfer discrete actions to continuous action
            action_continuous = trans_discrete_to_continuous(action, env, action_card)
            # The reason for converting action continuous to a single value list here is that Pendulum.step()
            # uses numpy.clip(), which requires the passed in action of the form of a list.
            next_state, reward, terminated, truncated, _ = env.step([action_continuous])
            # add transition to queue.
            replay_buffer.add(state, action, reward, next_state, terminated, truncated)
            # record the return of episode
            episode_return += reward
            if terminated or truncated:
                break
            state = next_state
            # if the number of transition of replay buffer is larger or equal to minimal_size, we start to optimize
            # the params of neural network.
            if replay_buffer.size() > HyperParams.minimal_size:
                transitions = replay_buffer.sample_batch(HyperParams.batch_size)
                # use above transitions to calculate loss function
                agent.update(transitions)
        if (i + 1) % 10 == 0:
            print("episodes:{}->{}, episode_returns_mean:{}.".format(i - 9, i, mean(returns[-10:])))
        returns.append(episode_return)
    env.close()

    # plot
    figure(dpi=400)
    plot(returns, c="darkblue")
    xlabel("episode")
    ylabel("return")
    show()

    env = gym.make("Pendulum-v1", render_mode="human")

    for i in range(10):
        state = env.reset(seed=0)[0]
        while True:
            env.render()
            action = agent.take_action(state)
            action_continuous = trans_discrete_to_continuous(action, env, action_card)
            next_state, reward, terminated, truncated, _ = env.step([action_continuous])
            print("transition: state:{}, action:{}, next_state:{}, reward:{}, terminated:{}, truncated:{}".format(state, action, next_state, reward, terminated, truncated))
            if terminated or truncated:
                break
            state = next_state