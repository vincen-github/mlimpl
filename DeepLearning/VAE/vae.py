from os import getcwd, makedirs
from os.path import exists

import torch
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot, savefig, close, axis, imshow
from scipy.io import loadmat
from torch import from_numpy, normal
from torch.nn import Module, Sequential, Linear, ReLU, BCELoss, Sigmoid
from torch.optim import Adam
from torch.utils.data import DataLoader


class VAE(Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()
        # representation dimension
        self.z_dim = z_dim

        # encoder whose output is mean.
        self.mu_encoder = Sequential(Linear(560, 200),
                                     ReLU(),
                                     Linear(200, self.z_dim),
                                     ReLU())
        # encoder whose output is sigma
        self.logvar_encoder = Sequential(Linear(560, 200),
                                         ReLU(),
                                         Linear(200, self.z_dim),
                                         ReLU())
        # In this example, we suppose posterior is N(μ,diag(σ)I).
        # decoder
        self.decoder = Sequential(Linear(z_dim, 200),
                                  ReLU(),
                                  Linear(200, 560),
                                  Sigmoid())

    def encode(self, x):
        mu = self.mu_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    # Sampling from q(z|x) will lead to high variance because the gradient of ELOB include ▽log(q(z|x)).
    # To reduce it, use trick named reparameterization.

    # But a valuable nation is that it is not necessary to use this trick as we can decompose ELOB
    def reparameterize(self, mu, logvar):
        # to KLD and reconstruction loss, not only that, we can analytically compute KLD and use BCE to replace
        # the other term.
        std = logvar.mul(0.5).exp()
        eps = normal(mean=0, std=1, size=std.size()).cuda()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 560))
        # not necessary
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    # hyper parameters
    EPOCH = 200
    BATCH_SIZE = 100
    LR = 1e-5

    # load frey face data
    img_rows = 28
    img_cols = 20

    ff_path = getcwd()

    # reshape dataset to be (sample_size, channels, rows, cols)
    ff = loadmat(ff_path + "\\frey_rawface.mat")['ff'].T.reshape((-1, 1, img_rows, img_cols))
    # rescale and to tensor
    ff = from_numpy(ff.astype('float32') / 255)
    # print(ff.shape)

    reconstruction_function = BCELoss()
    reconstruction_function.size_average = False


    def loss_func(rex, x, mu, logvar):
        BCE = reconstruction_function(rex, x.reshape(rex.shape[0], -1))
        KLD = mu.pow(2).add(logvar.exp()).mul(-1).add(1).add(logvar).sum().mul(-0.5)
        return KLD + BCE


    # model
    vae = VAE(z_dim=20)

    vae.to("cuda:0")

    # training block
    train_loader = DataLoader(ff, BATCH_SIZE, shuffle=True)
    opt = Adam(vae.parameters(), lr=LR)

    vae.train()
    train_loss = 0
    for epoch in range(EPOCH):
        for step, batch in enumerate(train_loader):
            batch = batch.cuda()
            opt.zero_grad()
            rexbatch, mu, logvar = vae(batch)
            loss = loss_func(rexbatch, batch, mu, logvar)
            loss.backward(retain_graph=True)
            train_loss += loss.data
            opt.step()

            if step == len(train_loader.dataset) // BATCH_SIZE - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t"Minibatch" Loss: {:.6f}'.format(
                    epoch, (step + 1) * len(batch), len(train_loader.dataset),
                           100. * (step + 1) / len(train_loader),
                           loss.data / len(batch)))

                rexs = rexbatch.data.cpu().numpy()[:16]

        # check the reconstruction result every ten epochs.
        if epoch % 20 == 0:
            fig = figure(figsize=(4, 4))
            gs = GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)
            for i, rex in enumerate(rexs):
                ax = subplot(gs[i])
                axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                imshow(rex.reshape(28, -1), cmap='gray')

                if not exists('out/'):
                    makedirs('out/')

                savefig('out/rex.png', bbox_inches='tight')
                close(fig)

    print('====> Epoch: {} Total batch loss: {:.4f}, '.format(
        epoch, train_loss / len(train_loader.dataset)))

    torch.save(vae, './out/save.model')
