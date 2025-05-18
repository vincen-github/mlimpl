from .w_mse import WMSE
from .simclr import SimCLR
from .haochen22 import Haochen22
from .barlow_twins import BarlowTwins
from .vicreg import Vicreg
from .spectral import Spectral
from .swav import SwAV

METHOD_LIST = ["vicreg", "haochen22", "w_mse", "barlow_twins", "simclr", "spectral", "swav"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "w_mse":
        return WMSE
    elif name == "contrastive":
        return Contrastive
    elif name == "barlow_twins":
        return BarlowTwins
    elif name == "haochen22":
        return Haochen22
    elif name == "vicreg":
        return Vicreg
    elif name == "spectral":
        return Spectral
    elif name == "simclr":
        return SimCLR
    elif name == "swav":
        return SwAV
