from os import getcwd

from scipy.io import loadmat
from torch import from_numpy, normal
from torch.nn import Module, Sequential, Linear, ReLU, BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader


class VAE(Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.z_dim = z_dim

        self.mu_encoder = Sequential(Linear(560, z_dim),
                                     ReLU())
        self.logvar_encoder = Sequential(Linear(560, z_dim),
                                         ReLU())
        self.decoder = Sequential(Linear(z_dim, 560),
                                  ReLU())

    def encode(self, x):
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x).exp_()
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparametrize(self, mu, logvar):
        std = logvar.exp_().sqrt_()
        eps = normal(mean=mu, std=std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # 前项传播过程
        mu, logvar = self.encode(x.view(-1, 560))

        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_func(self, x, rex, mu, logvar):
        BCE = BCELoss(rex, x.reshape(rex.shape[0], -1))
        KLD = mu.pow(2).add(logvar.exp()).mul(-1).add(1).add(logvar).sum().mul(-0.5)

        return BCE + KLD


if __name__ == "__main__":
    # hyper parameters
    EPOCH = 10
    BATCH_SIZE = 100
    LR = 1e-4

    # load frey face data
    img_rows = 28
    img_cols = 20

    ff_path = getcwd()

    # reshape dataset to be (sample_size, channels, rows, cols)
    ff = loadmat(ff_path + "\\frey_rawface.mat")['ff'].T.reshape((-1, 1, img_rows, img_cols))
    # rescale and to tensor
    ff = from_numpy(ff.astype('float32') / 255)
    # print(ff.shape)

    # model
    vae = VAE(z_dim=20)

    vae.to("cuda:0")

    # training block
    train_loader = DataLoader(ff, BATCH_SIZE, shuffle=True)
    opt = Adam(vae.parameters(), lr=LR)

    train_loss = 0
    for epoch in range(EPOCH):
        for step, batch in enumerate(train_loader):
            batch = batch.cuda()
            opt.zero_grad()
            rebatch, mu, logvar = vae(batch)
            loss = vae.loss_func(rebatch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.data[0]
            opt.step()

            if step == len(train_loader.dataset) / BATCH_SIZE - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t"Minibatch" Loss: {:.6f}'.format(
                    epoch, (step + 1) * len(batch), len(train_loader.dataset),
                           100. * (step + 1) / len(train_loader),
                           loss.data[0] / len(batch)))
