from torch import norm, cat, mean, matmul, triu, tril
from numpy import sqrt
import torch.nn as nn
import torch.nn.functional as F 
from .base import BaseMethod

def D(x0, x1, mu=1.0):
    mask0 = (norm(x0, p=2, dim=1) < sqrt(mu)).float().unsqueeze(1)
    mask1 = (norm(x1, p=2, dim=1) < sqrt(mu)).float().unsqueeze(1)
    z0 = mask0 * x0 + (1-mask0) * F.normalize(x0, dim=1) * sqrt(mu)
    z1 = mask1 * x1 + (1-mask1) * F.normalize(x1, dim=1) * sqrt(mu)
    loss_part1 = -2 * mean(z0 * z1) * z0.shape[1]
    square_term = matmul(z0, z1.T) ** 2
    loss_part2 = mean(triu(square_term, diagonal=1) + tril(square_term, diagonal=-1)) * \
                 z0.shape[0] / (z0.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu

class Spectral(BaseMethod):
    """ implements our ssl loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_f = D

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

