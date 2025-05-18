from torch import mm, no_grad, softmax, mean, log, exp, sum, zeros, ones, cat
from .base import BaseMethod
from torch.nn.functional import normalize

def swav_loss(x0, x1, C, temp=0.1): # temp = 0.1
    # C：prototypes (DxK)
    # temp: temperature
    N = x0.size(0)
    D = x0.size(1)

    x0 = normalize(x0)
    x1 = normalize(x1)

    scores0 = mm(x0, C) # N x K
    scores1 = mm(x1, C) # N x K
    
    with no_grad():
        q0 = sinkhorn(scores0)
        q1 = sinkhorn(scores1)
    
    p0 = softmax(scores0 / temp, dim=1)
    p1 = softmax(scores1 / temp, dim=1)
    loss = - 0.5 * mean(q0 * log(p1) + q1 * log(p0))

    return loss

def sinkhorn(scores, eps=0.05, niters=3): # eps = 0.05
    Q = exp(scores / eps).T
    Q /= sum(Q)
        
    K, N = Q.shape
    u, r, c = zeros(K).to("cuda"), ones(K).to("cuda") / K, ones(N).to("cuda") / N
    for _ in range(niters):
        u = sum(Q, dim=1)
        
        Q *= (r / u).unsqueeze(1)
        Q *= (c / sum(Q, dim=0)).unsqueeze(0)
    return (Q / sum(Q, dim=0, keepdim=True)).T

class SwAV(BaseMethod):
    """Implements SwAV loss"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_f = swav_loss

    def forward(self, samples, C):
        bs = len(samples[0])
        h = [self.model(x.cuda(non_blocking=True)) for x in samples]
        h = self.head(cat(h))
        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                x0 = h[i * bs: (i + 1) * bs]
                x1 = h[j * bs: (j + 1) * bs]
                loss += self.loss_f(x0, x1, C)
        loss /= self.num_pairs
        return loss
