from . import utils
from ._algorithms import Algorithms
from .rl_toolbox import RLToolbox


def algorithms(env):
    return Algorithms(env)


__all__ = ["Algorithms", "RLToolbox", "algorithms", "utils"]
