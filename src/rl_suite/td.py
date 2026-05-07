"""Temporal-difference control algorithms."""

from rl_suite._algorithms._base._td_base import _TDBaseAgent
from rl_suite._algorithms._td._td_algos._expected_sarsa import _ExpectedSarsa
from rl_suite._algorithms._td._td_algos._q_learning import _QLearning
from rl_suite._algorithms._td._td_algos._sarsa import _Sarsa
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
