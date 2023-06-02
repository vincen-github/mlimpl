from torch import log, tensor, float32
from torch.distributions import Categorical
from torch.optim import Adam

from ReinforcementLearning.PolicyGradient.PolicyNetwork import PolicyNetwork


class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_card, lr, gamma, device):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_card = action_card
        self.gamma = gamma
        self.device = device

        self.policy_net = PolicyNetwork(self.state_dim, self.hidden_dim, self.action_card).to(self.device)

        self.optimizer = Adam(self.policy_net.parameters(), lr=lr)

    def take_action(self, state):
        probs = self.policy_net(tensor(state).to(self.device))
        categorical = Categorical(probs)
        return categorical.sample().item()

    def update(self, transitions):
        G = 0
        self.optimizer.zero_grad()
        for state, action, next_state, reward, done in reversed(transitions):
            G = self.gamma * G + reward.to(self.device)
            loss = -log(self.policy_net(state.to(self.device))[action]) * G
            # Here utilizes the accumulation of gradient when backward to calculate
            # Σ_{t = 1}^T G_t▽logπ(a_t|s_t)
            loss.backward()
        self.optimizer.step()
