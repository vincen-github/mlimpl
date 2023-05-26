from torch.nn import Sequential, Linear, ReLU, Module


class Qnet(Module):
    def __init__(self, state_dim, hidden_dim, action_card):
        super(Qnet, self).__init__()
        self.model = Sequential(Linear(state_dim, hidden_dim),
                                ReLU(),
                                Linear(hidden_dim, action_card),
                                ReLU())

    def forward(self, x):
        return self.model(x)
