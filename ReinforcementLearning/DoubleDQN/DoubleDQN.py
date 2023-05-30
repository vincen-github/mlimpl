from numpy.random import random, randint

from torch import tensor, float32, empty, mean, int64, argmax
from torch.nn.functional import mse_loss
from torch.optim import Adam

from ReinforcementLearning.DoubleDQN.Qnet import Qnet


class DoubleDQN:
    def __init__(self, state_dim, hidden_dim, action_card, lr, gamma, eps, target_update_frequency, device="cuda:0"):
        self.action_card = action_card
        """
        params:
            1. state_dim: the dimension of single data from state space, note that u need to distinguish
                this notion to the cardinality of state space.
            2. hidden_dim: the dimension of hidden layer of neural network which is used to approximate Q*.
            3. action_card: the cardinality of action space. DQN only can handle the case that the dim of action space 
                is finite.
            4. lr: learning rate.
            5. gamma: discount factor.
            6. eps: param of ε-greedy.
            7. target_update_frequency: update frequency of target network.
            8. device: use gpu or cpu to train network.
        """

        self.state_dim = state_dim
        self.gamma = gamma
        self.eps = eps
        self.target_update_frequency = target_update_frequency
        self.count = 0  # record the number of update of Q*
        self.device = device

        self.q_net = Qnet(self.state_dim, hidden_dim, self.action_card).to(device)
        self.target_net = Qnet(self.state_dim, hidden_dim, self.action_card).to(device)

        # optimizer
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)

    def take_action(self, state):
        # note that we can't write policy π as a tabular as the cardinality of state space in infinite.
        if random() < self.eps:
            action = randint(self.action_card)
        else:
            # the state is originally in cpu, move it to gpu.
            state = tensor(state, dtype=float32).to(self.device)
            # Tensor.item() → number
            # Returns the value of this tensor as a standard Python number.
            # This only works for tensors with one element.
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

        # loss = 1/2nΣ[r + max_{a'}target_Q(s', a') - Q(s,a)]

        double_dqn_loss = mean(mse_loss(
            self.q_net(states).gather(1, actions.view(-1, 1)).flatten(),
            (rewards + self.gamma * self.target_net(next_states).gather(1, argmax(self.q_net(states), 1).view(-1, 1)).flatten() * (1 - terminateds) * (1 - truncateds))
        ))
        # optimize
        self.optimizer.zero_grad()
        double_dqn_loss.backward()
        self.optimizer.step()

        # Regularly update the parameters of the target network.
        if self.count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Count record the number of parameter updates for q_net
        self.count += 1
