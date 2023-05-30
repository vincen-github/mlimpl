from matplotlib.pyplot import plot, show, figure, xlabel, ylabel

from ReinforcementLearning.DoubleDQN.DoubleDQN import DoubleDQN
from ReinforcementLearning.DoubleDQN.ReplayBuffer import ReplayBuffer
from ReinforcementLearning.DoubleDQN import HyperParams
import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Environment Reference: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
    # get instance of Class ReplayBuffer
    replay_buffer = ReplayBuffer(HyperParams.buffer_capacity)

    state_dim = env.observation_space.shape[0]
    action_card = env.action_space.n

    agent = DoubleDQN(state_dim, HyperParams.hidden_dim, action_card, HyperParams.lr, HyperParams.gamma, HyperParams.eps,
                HyperParams.target_update_frequency, HyperParams.device)

    returns = []

    for i in range(HyperParams.num_episodes):
        episode_return = 0
        state = env.reset()[0]
        while True:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
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
            print("episode:{}, episode_return:{}.".format(i, episode_return))
        returns.append(episode_return)
    env.close()

    # plot
    figure(dpi=400)
    plot(returns, c="darkblue")
    xlabel("eposide")
    ylabel("return")
    show()

    env = gym.make("CartPole-v1", render_mode="human")

    for i in range(10):
        state = env.reset()[0]
        while True:
            env.render()
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            state = next_state