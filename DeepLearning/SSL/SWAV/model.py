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
    model = getattr(models, arch)(weights=None)
    if dataset != "imagenet":
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if dataset == "cifar10" or dataset == "cifar100":
        model.maxpool = nn.Identity()
    out_size = model.fc.in_features
    model.fc = nn.Identity()

    return nn.DataParallel(model), out_size
