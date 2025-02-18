from torch import mm, eye, cat, norm
from .base import BaseMethod

def haochen22(x0, x1, lambda_param=1):
    N = x0.size(0)
    D = x0.size(1)

    c0 = x0.T @ x0 / N # DxD
    c1 = x1.T @ x1 / N
    c_diff = (1 / 2 * c0 + 1 / 2 * c1 - eye(D, device=c0.device)).pow(2)
    return norm(x0 - x1, p=2, dim=1).pow(2).mean() + lambda_param * c_diff.sum()

class Haochen22(BaseMethod):
    """ implements our ssl loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_f = haochen22

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
