from copy import deepcopy
from .config_mindspore import cfg as defaultcdg

def get_default_cfg():
    return deepcopy(defaultcdg)