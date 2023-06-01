from numpy.random import random, randint

from torch import float32, tensor, empty, int64
from torch.nn.functional import mse_loss
from torch.optim.adam import Adam

from ReinforcementLearning.DuelingNetwork.Networks import VANet


class DuelingNetwork:
    """
    Dueling network is another improved algorithm for dqn, which decompose the action value function to three parts that
    Q(s,a) = V(s) + A(s,a) + maxA(s,a) or Q(s,a) = V(s) + A(s,a) + mean{A(s,a)}.
    Apart from that, we preserve the techniques that target network, TD target of double dnq in this implementation.
    """
    def __init__(self, state_dim, vnet_hidden_dim, anet_hidden_dim, action_card, lr, gamma, eps,
                 target_update_frequency, device="cuda:0", update_mode="mean", share=True):
        self.state_dim = state_dim
        self.vnet_hidden_dim = vnet_hidden_dim
        self.anet_hidden_dim = anet_hidden_dim
        self.action_card = action_card

        self.gamma = gamma
        self.eps = eps
        self.target_update_frequency = target_update_frequency
        self.count = 0
        self.device = device
        self.update_mode = update_mode
        self.share = share

        self.q_net = VANet(self.state_dim, self.vnet_hidden_dim, self.anet_hidden_dim, self.action_card,
                           self.update_mode, self.share).to(self.device)
        self.target_net = VANet(self.state_dim, self.vnet_hidden_dim, self.anet_hidden_dim, self.action_card,
                           self.update_mode, self.share).to(self.device)

        self.optimizer = Adam(self.q_net.parameters(), lr)

    def take_action(self, state):
        if random() < self.eps:
            action = randint(self.action_card)
        else:
            state = tensor(state, dtype=float32).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transitions):
        """
        Given transitions, update the parameters of neural network.
        """
        batch_size = len(transitions)
        states, actions, rewards, next_states, terminateds, truncateds = empty((batch_size, self.state_dim),
            device=self.device), empty(batch_size, dtype=int64, device=self.device), empty(batch_size, dtype=int64,
            device=self.device), empty((batch_size, self.state_dim), device=self.device), empty(batch_size, dtype=int64,
            device=self.device), empty(batch_size, dtype=int64, device=self.device)
        for i, transition in zip(range(batch_size), transitions):
            states[i], actions[i], rewards[i], next_states[i], terminateds[i], truncateds[i] = tensor(
                transition[0]), tensor(transition[1]), tensor(transition[2]), tensor(transition[3]), tensor(
                transition[4]), tensor(transition[5])

        # loss = 1/2nΣ[r + max_{a'}target_Q(s', argmax{a'}Q(s',a')) - Q(s,a)]
        q_net_max_actions = self.q_net(next_states).max(1)[1].view(-1, 1)
        # q_net_max_actions_requires_grad = False
        double_dqn_loss = mse_loss(
            self.q_net(states).gather(1, actions.view(-1, 1)).flatten(),
            (rewards + self.gamma * self.target_net(next_states).gather(1, q_net_max_actions).flatten() * (
                        1 - terminateds) * (1 - truncateds))
        )
        # optimize
        self.optimizer.zero_grad()
        double_dqn_loss.backward()
        self.optimizer.step()

        # Regularly update the parameters of the target network.
        if self.count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Count record the number of parameter updates for q_net
        self.count += 1