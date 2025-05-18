from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .tiny_in import TinyImageNet


DS_LIST = ["cifar10", "cifar100", "tiny_in"]


def get_ds(name):
    assert name in DS_LIST
    if name == "cifar10":
        return CIFAR10
    elif name == "cifar100":
        return CIFAR100
    elif name == "tiny_in":
        return TinyImageNet
