from torch.nn import Sequential, Linear, ReLU, Softmax, Module


class PolicyNet(Module):
    def __init__(self, state_dim, hidden_dim, action_card):
        super(PolicyNet, self).__init__()
        self.model = Sequential(Linear(state_dim, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, action_card),
                                Softmax(dim=0))

    def forward(self, x):
        return self.model(x)


class ValueNet(Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.model = Sequential(Linear(state_dim, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, 1))

    def forward(self, x):
        return self.model(x)
