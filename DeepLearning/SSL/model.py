import torch.nn as nn
from torchvision import models


def get_head(out_size, cfg):
    """ creates projection head g() from config """
    x = []
    in_size = out_size
    for _ in range(cfg.head_layers - 1):
        x.append(nn.Linear(in_size, cfg.head_size))
        if cfg.add_bn:
            x.append(nn.BatchNorm1d(cfg.head_size))
        x.append(nn.ReLU())
        in_size = cfg.head_size
    x.append(nn.Linear(in_size, cfg.emb))
    return nn.Sequential(*x)


def get_model(arch, dataset):
    """ creates encoder E() by name and modifies it for dataset """
    model = getattr(models, arch)(weights="DEFAULT")
    if dataset != "imagenet":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if dataset == "cifar10" or dataset == "cifar100":
        model.maxpool = nn.Identity()
    out_size = model.fc.in_features
    model.fc = nn.Identity()

    return nn.DataParallel(model), out_size

class Critic(nn.Module):
    def __init__(self, ndim):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            self._block(ndim, 128),
            self._block(128, ndim),
            nn.Linear(ndim, 1)
        )
        
    
    def _block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
#            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, X):
        validity = self.model(X)
        return validity
