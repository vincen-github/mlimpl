from torch import unsqueeze, FloatTensor, tanh, sigmoid, zeros, tensor, normal, Tensor, concat, float32
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

# hyper parameters
epoch = 1
batch_size = 32
time_step = 28
input_size = 28
LR = 0.01
DOWNLOAD_MNIST = False

train = FashionMNIST(root="../dataset/", train=True, transform=ToTensor(), download=False)
test = FashionMNIST(root="../dataset/", train=False, transform=ToTensor(), download=False)

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)

test_x = unsqueeze(test.data, dim=1).type(FloatTensor)[:2000] / 255
test_y = test.targets[:2000]


class LSTM(Module):
    def __init__(self, c0, h0, Wi, Wf, Wo, Wg):
        super(Module, self).__init__()
        self.Wi = Wi
        self.Wf = Wf
        self.Wo = Wo
        self.Wg = Wg
        self.c = c0
        self.h = h0

    def forward(self, x):
        i = sigmoid(self.Wi @ compose)
        f = sigmoid(self.Wf @ compose)
        o = sigmoid(self.Wo @ compose)
        g = tanh(self.Wg @ compose)

        self.c = f * self.c + i * g
        h = o * tanh(self.c)


def lstm(X: Tensor, h0=zeros(10, dtype=float32),
         c0=zeros(10, dtype=float32)):
    """
    h_size is equal to row number of Wi, Wf, Wo, Wg.
    """
    W_nrow = h0.size
    W_ncol = X.shape[0] + W_nrow

    print(W_nrow)
    # initialize parameters
    Wi = tensor(normal(mean=0, std=1, size=(W_nrow, W_ncol)), requires_grad=True)
    Wf = tensor(normal(mean=0, std=1, size=(W_nrow, W_ncol)), requires_grad=True)
    Wo = tensor(normal(mean=0, std=1, size=(W_nrow, W_ncol)), requires_grad=True)
    Wg = tensor(normal(mean=0, std=1, size=(W_nrow, W_ncol)), requires_grad=True)

    def forward(x, h, c):
        compose = concat([h, x])
        i = sigmoid(Wi @ compose)
        f = sigmoid(Wf @ compose)
        o = sigmoid(Wo @ compose)
        g = tanh(Wg @ compose)

        c = f * c + i * g
        h = o * tanh(c)
        return h, c

    h, c = forward(x, h0, c0)

    Wi.backward()
    Wf.backward()
    Wo.backward()
    Wg.backward()
