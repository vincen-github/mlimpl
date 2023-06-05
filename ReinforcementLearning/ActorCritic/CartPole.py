import gym
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
from numpy import mean
from torch import tensor

from ReinforcementLearning.ActorCritic import HyperParams
from ReinforcementLearning.ActorCritic.ActorCritic import ActorCritic

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_card = env.action_space.n
    agent = ActorCritic(state_dim, HyperParams.hidden_dim, action_card, HyperParams.actor_lr, HyperParams.critic_lr,
                        HyperParams.gamma, HyperParams.target_update_frequency, HyperParams.device)

    returns = []

    for i in range(HyperParams.num_episodes):
        transitions = []
        # record episode's return to plot
        episode_return = 0
        state = env.reset()[0]
        while True:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            transitions.append(
                (tensor(state), tensor(action), tensor(next_state), tensor(reward), terminated or truncated))
            if terminated or truncated:
                returns.append(episode_return)
                break
            state = next_state
        agent.update(transitions)
        if (i + 1) % 50 == 0:
            print("episodes:{}->{}, episode_returns_mean:{}.".format(i - 49, i, mean(returns[i - 49:i])))
    env.close()

    # plot
    figure(dpi=400)
    plot(returns, c="darkblue")
    xlabel("episode")
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
