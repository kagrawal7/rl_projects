"""Single public facade for rl_suite.

Import this module to access algorithms and utilities:

    import rl_suite.RLToolbox as rl_var
"""

from ._algorithms import _Algorithms
from ._utils import _Utilities


algorithms = _Algorithms()
utils = _Utilities()


__all__ = ["algorithms", "utils"]
