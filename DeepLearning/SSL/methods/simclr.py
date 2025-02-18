from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseMethod


def simclr(x0, x1, tau, norm):
    # https://github.com/google-research/simclr/blob/master/objective.py
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    if norm:
        x0 = F.normalize(x0, p=2, dim=1)
        x1 = F.normalize(x1, p=2, dim=1)
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits11 = x1 @ x1.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    logits10 = x1 @ x0.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2


class SimCLR(BaseMethod):
    """ implements contrastive loss https://arxiv.org/abs/2002.05709 """

    def __init__(self, cfg):
        """ init additional BN used after head """
        super().__init__(cfg)
        self.bn_last = nn.BatchNorm1d(cfg.emb)
        self.loss_f = partial(simclr, tau=cfg.tau, norm=cfg.norm)

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.bn_last(self.head(torch.cat(h)))
        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs : (i + 1) * bs]
                x1 = h[j * bs : (j + 1) * bs]
                loss += self.loss_f(x0, x1)
        loss /= self.num_pairs
        return loss
