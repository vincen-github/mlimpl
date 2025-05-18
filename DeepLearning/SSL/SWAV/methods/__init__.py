from .swav import SwAV

METHOD_LIST = ["swav"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "swav":
        return SwAV
