"""Flat access to all implemented reinforcement-learning algorithms."""

from rl_suite import dp, mc, td
from rl_suite._discovery import class_names


_FAMILY_MODULES = (dp, mc, td)


def _algorithm_classes() -> list[type]:
    classes = []
    for module in _FAMILY_MODULES:
        classes.extend(module._algorithm_classes())
    return classes


for _algorithm in _algorithm_classes():
    globals()[_algorithm.__name__] = _algorithm

OffPolicyMC = mc.OffPolicyMC


def get_implemented_algorithms() -> list[str]:
    """Return all public algorithm classes exposed by rl_suite."""
    return class_names(_algorithm_classes())


get_implemented_algs = get_implemented_algorithms


__all__ = [
    *get_implemented_algorithms(),
    "OffPolicyMC",
    "get_implemented_algs",
    "get_implemented_algorithms",
]
