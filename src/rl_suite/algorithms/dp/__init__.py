"""Dynamic-programming algorithms."""

from rl_suite.algorithms._base.dynamic_programming import _GPIBaseAgent
from rl_suite.algorithms.dp.policy_iteration import PolicyIteration as PolicyIteration
from rl_suite.algorithms.dp.stochastic_policy_evaluation import (
    StochasticPolicyEvaluation as StochasticPolicyEvaluation,
)
from rl_suite.algorithms.dp.value_iteration import ValueIteration as ValueIteration
from rl_suite._discovery import class_names, discover_algorithm_classes


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
