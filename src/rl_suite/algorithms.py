"""Public algorithm namespaces.

Algorithms are intentionally not bound to a toolbox instance. Pass the
environment to the algorithm constructor you want to run:

    import rl_suite.algorithms as algs
    agent = algs.td.sarsa(env)
"""

from ._algorithms._gpi._gpi import _GPI
from ._algorithms._mc._mc import _MC
from ._algorithms._td._td import _TD


gpi = _GPI()
mc = _MC()
td = _TD()


__all__ = ["gpi", "mc", "td"]
