from . import utils
from ._algorithms import Algorithms
from ._utils import _Utilities

class RLToolbox:

    def __init__(self):
        self.algorithms = Algorithms()
        self.utils = _Utilities()
