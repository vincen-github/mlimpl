from torch import tensor, log
from torch.distributions import Categorical
from torch.optim import Adam

from ReinforcementLearning.ActorCritic.Networks import PolicyNet, ValueNet


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, target_update_frequency, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        # I don't understand why the trick that dualing network will lead the fail of training?
        # self.target_net = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        # self.target_update_frequency = target_update_frequency
        self.device = device

        self.count = 0

    def take_action(self, state):
        probs = self.actor(tensor(state).to(self.device))
        categorical = Categorical(probs)
        return categorical.sample().item()

    def update(self, transitions):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        for state, action, next_state, reward, done in transitions:
            state = state.to(self.device)
            td_target = reward.to(self.device) + self.gamma * self.critic(next_state.to(self.device)) * (1 - done)
            # δ_t = r_t + γ V_ω-(s_{t+1}) - V_ω(s_t) = td_target - V_ω(s_t)
            critic_loss = pow(td_target - self.critic(state), 2)
            # Σ_{t = 1}^T δ_t▽logπ(a_t|s_t)
            delta = td_target - self.critic(state)
            # take negative is as we need to perform gradient ascent instead of descent.
            actor_loss = -log(self.actor(state)[action]) * delta.detach()
            # Here utilizes the accumulation of gradient when backward to calculate
            actor_loss.backward()
            critic_loss.backward()
        # It seems that the training will be more unstable if we exchange the order of these two steps of optimization.
        # Why?
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        # # Regularly update the parameters of the target network.
        # if self.count % self.target_update_frequency == 0:
        #     self.target_net.load_state_dict(self.critic.state_dict())
        #
        # self.count += 1
