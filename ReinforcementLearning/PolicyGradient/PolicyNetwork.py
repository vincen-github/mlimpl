from torch.nn import Module, Sequential, Linear, ReLU, Softmax


class PolicyNetwork(Module):
    def __init__(self, state_dim, hidden_dim, action_card):
        super(PolicyNetwork, self).__init__()
        self.model = Sequential(Linear(state_dim, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, action_card),
                                Softmax(dim=0))

    def forward(self, x):
        return self.model(x)
