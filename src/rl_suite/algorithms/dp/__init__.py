"""Dynamic-programming algorithms."""

from rl_suite.algorithms.base.dynamic_programming import _GPIBaseAgent
from rl_suite.algorithms.dp.policy_iteration import _Deterministic
from rl_suite.algorithms.dp.stochastic_policy_evaluation import _Stochastic
from rl_suite.algorithms.dp.value_iteration import _ValueIteration
from rl_suite._discovery import class_names, discover_algorithm_classes


class PolicyIteration(_Deterministic):
    """Policy iteration with a deterministic policy table."""


class StochasticPolicyEvaluation(_Stochastic):
    """Policy evaluation with a stochastic policy table."""


class ValueIteration(_ValueIteration):
    """Value iteration for finite Markov decision processes."""


def _algorithm_classes() -> list[type]:
    return discover_algorithm_classes(__name__, _GPIBaseAgent)


def get_implemented_algorithms() -> list[str]:
    """Return the implemented dynamic-programming algorithms."""
    return class_names(_algorithm_classes())


get_implemented_algs = get_implemented_algorithms


__all__ = [
    *get_implemented_algorithms(),
    "get_implemented_algs",
    "get_implemented_algorithms",
]
