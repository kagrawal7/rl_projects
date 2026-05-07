"""Algorithm namespaces exposed through ``rl_suite.RLToolbox``."""

from rl_suite._algorithms import _Algorithms


_algorithms = _Algorithms()

gpi = _algorithms.gpi
mc = _algorithms.mc
td = _algorithms.td


__all__ = ["gpi", "mc", "td"]
