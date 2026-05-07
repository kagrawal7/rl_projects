"""Temporal-difference control algorithms."""

from rl_suite.algorithms._base.temporal_difference import _TDBaseAgent
from rl_suite.algorithms.td.q_learning import QLearning as QLearning
from rl_suite.algorithms.td.sarsa import SARSA as SARSA
from rl_suite.algorithms.td.expected_sarsa import ExpectedSARSA as ExpectedSARSA
from rl_suite._discovery import class_names, discover_algorithm_classes


def _algorithm_classes() -> list[type]:
    return discover_algorithm_classes(__name__, _TDBaseAgent)


def get_implemented_algorithms() -> list[str]:
    """Return the implemented temporal-difference algorithms."""
    return class_names(_algorithm_classes())


get_implemented_algs = get_implemented_algorithms


__all__ = [
    *get_implemented_algorithms(),
    "get_implemented_algs",
    "get_implemented_algorithms",
]
