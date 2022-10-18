from math import sqrt

from matplotlib.pyplot import plot, show, legend
from torch import tanh, sigmoid, FloatTensor, Tensor, bmm, zeros, normal, argmax, ones
from torch.nn import Module, Parameter
from torch.nn import init
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


class LSTM(Module):
    """
    lstm is a common neural network which is to tackle serialized data. Compared with RNN, It can handle vanishing
    gradient problem by preserving long-term memory.

    Reference:
    1. https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
    2. 2022 Summer Short Course in TianYuan Mathematical Center in Central China:
            Mathematical Theory and Applications of Deep Learning
            Prof. Haizhao Yang (University of Maryland,CollegePark)
            Course Video :Course(2) replay in http://tmcc.whu.edu.cn/info/1262/2052.htm
    3. https://www.bilibili.com/video/BV1zq4y1m7aH?spm_id_from=333.337.search-card.all.click
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(LSTM, self).__init__()
        self.h = None
        self.c = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # i's (input gate) parameters
        self.w_hi = Parameter(Tensor(self.hidden_size, self.hidden_size))
        self.w_xi = Parameter(Tensor(self.hidden_size, self.input_size))
        self.b_hi = Parameter(Tensor(self.hidden_size, 1))
        self.b_xi = Parameter(Tensor(self.hidden_size, 1))

        # f's (forget gate) parameters
        self.w_hf = Parameter(Tensor(self.hidden_size, self.hidden_size))
        self.w_xf = Parameter(Tensor(self.hidden_size, self.input_size))
        self.b_hf = Parameter(Tensor(self.hidden_size, 1))
        self.b_xf = Parameter(Tensor(self.hidden_size, 1))

        # o's (output gate) parameters
        self.w_ho = Parameter(Tensor(self.hidden_size, self.hidden_size))
        self.w_xo = Parameter(Tensor(self.hidden_size, self.input_size))
        self.b_ho = Parameter(Tensor(self.hidden_size, 1))
        self.b_xo = Parameter(Tensor(self.hidden_size, 1))

        # g's (gate gate) parameters
        self.w_hg = Parameter(Tensor(self.hidden_size, self.hidden_size))
        self.w_xg = Parameter(Tensor(self.hidden_size, self.input_size))
        self.b_hg = Parameter(Tensor(self.hidden_size, 1))
        self.b_xg = Parameter(Tensor(self.hidden_size, 1))

        self.w = Parameter(Tensor(self.output_size, self.hidden_size))

        self.init_param()

    def init_param(self):
        """
        reset weights
        """
        stdv = 1.0 / sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, X, c0, h0):
        """
        x's shape = (batch, seq, feature)
        """
        batch_size, seq_size, _ = X.shape

        # original c's size : (hidden_size, 1), h's size: (hidden_size, 1). To facilitate the following calculation,
        # we need to reshape it to be (batch_size, hidden_size, 1).
        self.c, self.h = c0.unsqueeze(0).tile(batch_size, 1, 1), h0.unsqueeze(0).tile(batch_size,
                                                                                      1, 1)

        # output of lstm is equal to (output, cn, hn), the size of output on LHS is (batch_size, seq_size, hidden_size)
        # In this application, we do not need to output this feature.
        # self.output = zeros(self.batch_size, self.seq_size, self.hidden_size)

        # reshape w' shape as (batch_size, hidden_size, input_size)
        batch_w_hi = self.w_hi.unsqueeze(0).tile(batch_size, 1, 1)
        batch_w_xi = self.w_xi.unsqueeze(0).tile(batch_size, 1, 1)
        batch_w_hf = self.w_hf.unsqueeze(0).tile(batch_size, 1, 1)
        batch_w_xf = self.w_xf.unsqueeze(0).tile(batch_size, 1, 1)
        batch_w_ho = self.w_ho.unsqueeze(0).tile(batch_size, 1, 1)
        batch_w_xo = self.w_xo.unsqueeze(0).tile(batch_size, 1, 1)
        batch_w_hg = self.w_hg.unsqueeze(0).tile(batch_size, 1, 1)
        batch_w_xg = self.w_xg.unsqueeze(0).tile(batch_size, 1, 1)

        batch_b_hi = self.b_hi.unsqueeze(0).tile(batch_size, 1, 1)
        batch_b_xi = self.b_xi.unsqueeze(0).tile(batch_size, 1, 1)
        batch_b_hf = self.b_hf.unsqueeze(0).tile(batch_size, 1, 1)
        batch_b_xf = self.b_xf.unsqueeze(0).tile(batch_size, 1, 1)
        batch_b_ho = self.b_ho.unsqueeze(0).tile(batch_size, 1, 1)
        batch_b_xo = self.b_xo.unsqueeze(0).tile(batch_size, 1, 1)
        batch_b_hg = self.b_hg.unsqueeze(0).tile(batch_size, 1, 1)
        batch_b_xg = self.b_xg.unsqueeze(0).tile(batch_size, 1, 1)

        for t in range(seq_size):
            # sequence data at time t
            x = X[:, t, :]
            # shape of x : (batch_size, input_size)
            # reshape x's size : (batch_size, input_size, 1)
            x = x.unsqueeze(-1)

            # (batch_w_hi @ h) 's shape: (batch_size, hidden_size, 1)
            # bmm(self.w_xi, x)'s shape: (batch_size, hidden_size, 1)
            i = sigmoid(bmm(batch_w_hi, self.h) + batch_b_hi + bmm(batch_w_xi, x) + batch_b_xi)
            f = sigmoid(bmm(batch_w_hf, self.h) + batch_b_hf + bmm(batch_w_xf, x) + batch_b_xf)
            o = sigmoid(bmm(batch_w_ho, self.h) + batch_b_ho + bmm(batch_w_xo, x) + batch_b_xo)
            g = tanh(bmm(batch_w_hg, self.h) + batch_b_hg + bmm(batch_w_xg, x) + batch_b_xg)

            self.c = f * self.c + i * g
            self.h = o * tanh(self.c)

            # self.output[:, t, :] = self.h

        # out: (batch_size, output_size, 1), for i-th sample, we can get predict label distribution through out[i].
        # out[i]: (output_size, 1)
        out = bmm(self.w.unsqueeze(0).tile(batch_size, 1, 1), self.h)

        return out


if __name__ == "__main__":
    # hyper parameters
    EPOCH = 2
    BATCH_SIZE = 32
    HIDDEN_SIZE = 28
    INPUT_SIZE = 28
    OUTPUT_SIZE = 10
    LR = 1e-2

    # u can choose the way to initialize h0 and c0.
    # c0 = normal(0, 1, size=(HIDDEN_SIZE, 1))
    # h0 = normal(0, 1, size=(HIDDEN_SIZE, 1))
    c0 = ones(size=(HIDDEN_SIZE, 1))
    h0 = ones(size=(HIDDEN_SIZE, 1))

    train = FashionMNIST(root="../dataset/", train=True, transform=ToTensor(), download=True)
    test = FashionMNIST(root="../dataset/", train=False, transform=ToTensor(), download=True)

    # pick 2000 samples to speed up testing
    test_x = test.data.type(FloatTensor)[:2000] / 255.
    # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test.targets[:2000]

    lstm = LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = Adam(lstm.parameters(), lr=LR)
    loss_func = CrossEntropyLoss()

    train_err_ls = []
    test_acc_ls = []
    step_ls = []

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.view(-1, 28, 28)

            output = lstm(batch_x, c0, h0)
            loss = loss_func(output.squeeze(-1), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                train_err_ls.append(loss.detach().numpy())

                pred = argmax(lstm(test_x, c0, h0).squeeze(-1), dim=1)

                test_acc = (test_y == pred).sum().detach().numpy() / pred.shape[0]

                test_acc_ls.append(test_acc)

                print("EPOCH:{}, STEP:{}, TRAIN_ERROR:{}, TEST_ACC:{}".format(epoch, step + 1, loss, test_acc))
                # 1800 is the max_step.
                step_ls.append(epoch * 1800 + step)

    plot(step_ls, train_err_ls, label="train_err")
    plot(step_ls, test_acc_ls, label="test_acc")
    legend()
    show()
