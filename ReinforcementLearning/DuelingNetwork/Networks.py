from torch.nn import Module, Linear
from torch.nn.functional import relu


class VNet(Module):
    def __init__(self, state_dim, hidden_dim):
        super(VNet, self).__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(relu(self.fc1(x)))


class ANet(Module):
    def __init__(self, state_dim, hidden_dim, action_card):
        super(ANet, self).__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, action_card)

    def forward(self, x):
        return self.fc2(relu(self.fc1(x)))


class VANet(Module):
    def __init__(self, state_dim, vnet_hidden_dim, anet_hidden_dim, action_card, update_mode="mean", share=True):
        super(VANet, self).__init__()
        self.vnet = VNet(state_dim, vnet_hidden_dim)
        self.anet = ANet(state_dim, anet_hidden_dim, action_card)

        self.update_mode = update_mode

        self._share = share
        # share parameters of the first linear layer.
        if self._share:
            self.anet.fc1 = self.vnet.fc1
            # print(self.anet.fc1 is self.vnet.fc1)

    def forward(self, x):
        v = self.vnet(x)
        a = self.anet(x)
        if self.update_mode == "max":
            q = v + a - a.max()
        if self.update_mode == "mean":
            q = v + a - a.mean()
        return q
        @property
        def update_mode(self):
            return self._update_mode

        @update_mode.setter
        def update_mode(self, update_mode):
            if update_mode in ["max", "mean"]:
                self._update_mode = update_mode
            else:
                raise ValueError("Parameter update_mode must either be 'max' or 'mean'.")

        @property
        def share(self):
            return self._share

        @share.setter
        def share(self, share):
            if not isinstance(share, bool):
                raise TypeError("share must be bool...")
            if share and self.VNet_hidden_dim != self.ANet_hidden_dim:
                raise ValueError("VNet_hidden_dim must be equal to ANet_hidden_dim in the case that share == True...")
            else:
                self._share = share
