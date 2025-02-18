from torch.nn.functional import normalize, relu, mse_loss
from .whitening import Whitening2d
from .base import BaseMethod
from torch import empty_like, eye, norm, cat, randperm, trace, einsum

def variance(x, gamma=1):
    return relu(gamma - x.std(0)).mean()

def invariance(x0, x1):
    return mse_loss(x0, x1)

def covariance(x):
    n, d = x.shape
    mu = x.mean(0)
    cov = einsum("ni,nj->ij", x-mu, x-mu) / (n - 1)
    off_diag = cov.pow(2).sum() - cov.pow(2).diag().sum()
    return off_diag / d

def vicreg(x0, x1, lambda_param=1e-3):
    N = x0.size(0)
    D = x0.size(1)
    
    la, mu, nu = 25, 25, 1
    var0, var1 = variance(x0), variance(x1)
    inv = invariance(x0, x1)
    cov0, cov1 = covariance(x0), covariance(x1)

    loss = la * inv + mu * (var0 + var1) + nu * (cov0 + cov1)
    return loss

class Vicreg(BaseMethod):
    """ implements our ssl loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_f = vicreg

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(cat(h))
        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs: (i + 1) * bs]
                x1 = h[j * bs: (j + 1) * bs]
                loss += self.loss_f(x0, x1)
        loss /= self.num_pairs
        return loss
