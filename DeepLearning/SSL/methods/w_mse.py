import torch
import torch.nn.functional as F
from .whitening import Whitening2d
from .base import BaseMethod
from .norm_mse import norm_mse_loss


class WMSE(BaseMethod):
    """ implements W-MSE loss """

    def __init__(self, cfg):
        """ init whitening transform """
        super().__init__(cfg)
        self.whitening = Whitening2d(cfg.emb, eps=cfg.w_eps, track_running_stats=False)
        self.loss_f = norm_mse_loss if cfg.norm else F.mse_loss
        self.w_iter = cfg.w_iter
        self.w_size = cfg.bs if cfg.w_size is None else cfg.w_size

    def forward(self, samples):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(torch.cat(h))
        loss = 0
        for _ in range(self.w_iter):
            z = torch.empty_like(h)
            perm = torch.randperm(bs).view(-1, self.w_size)
            for idx in perm:
                for i in range(len(samples)):
                    z[idx + i * bs] = self.whitening(h[idx + i * bs])
            for i in range(len(samples) - 1):
                for j in range(i + 1, len(samples)):
                    x0 = z[i * bs : (i + 1) * bs]
                    x1 = z[j * bs : (j + 1) * bs]
                    loss += self.loss_f(x0, x1)
        loss /= self.w_iter * self.num_pairs
        return loss
