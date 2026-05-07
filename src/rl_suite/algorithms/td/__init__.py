"""Temporal-difference control algorithms."""

from rl_suite.algorithms.base.temporal_difference import _TDBaseAgent
from rl_suite.algorithms.td.expected_sarsa import _ExpectedSarsa
from rl_suite.algorithms.td.q_learning import _QLearning
from rl_suite.algorithms.td.sarsa import _Sarsa
from rl_suite._discovery import class_names, discover_algorithm_classes


class QLearning(_QLearning):
    """Tabular off-policy Q-learning control."""


class SARSA(_Sarsa):
    """Tabular on-policy SARSA control."""


class ExpectedSARSA(_ExpectedSarsa):
    """Tabular Expected SARSA control."""


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
