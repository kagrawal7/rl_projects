from ._algorithms import Algorithms
from ._utils import _Utilities


class RLToolbox:
    algorithms = Algorithms
    utils = _Utilities()

    def __init__(self, env=None, discretization: dict | None = None):
        self.env = None
        self.utils = self.__class__.utils
        if env is not None:
            self.set_env(env, discretization=discretization)

    def set_env(self, env, discretization: dict | None = None):
        self.env = env
        self.utils = _Utilities(env, discretization=discretization)
        return self
