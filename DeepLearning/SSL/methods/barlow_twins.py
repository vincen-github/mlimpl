from torch import mm, eye, cat
from .base import BaseMethod

def barlow_twins(x0, x1, lambda_param=1e-3):
    # normalize repr. along the batch dimension
    x0_norm = (x0 - x0.mean(0)) / x0.std(0) # NxD
    x1_norm = (x1 - x1.mean(0)) / x1.std(0) # NxD

    N = x0.size(0)
    D = x0.size(1)

    # cross-correlation matrix
    c = mm(x0_norm.T, x1_norm) / N # DxD
    # loss
    c_diff = (c - eye(D, device=x0.device)).pow(2) # DxD
    # multiply off-diagonal elems of c_diff by lambda
    c_diff[~eye(D, dtype=bool)] *= lambda_param
    loss = c_diff.sum()

    return loss

class BarlowTwins(BaseMethod):
    """ implements our ssl loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_f = barlow_twins

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
