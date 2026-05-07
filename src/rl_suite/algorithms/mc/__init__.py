"""Monte Carlo control algorithms."""

from rl_suite.algorithms.mc.off_policy import _OffPolicyAgent
from rl_suite._discovery import class_names, discover_algorithm_classes


class OffPolicyMonteCarlo(_OffPolicyAgent):
    """Weighted importance-sampling off-policy Monte Carlo control."""


OffPolicyMC = OffPolicyMonteCarlo


def _algorithm_classes() -> list[type]:
    return discover_algorithm_classes(__name__, _OffPolicyAgent)


def get_implemented_algorithms() -> list[str]:
    """Return the implemented Monte Carlo algorithms."""
    return class_names(_algorithm_classes())


get_implemented_algs = get_implemented_algorithms


__all__ = [
    *get_implemented_algorithms(),
    "OffPolicyMC",
    "get_implemented_algs",
    "get_implemented_algorithms",
]
