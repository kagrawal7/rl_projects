"""Dynamic-programming algorithm family facade."""

from rl_suite.algos._dynamic_programming import (
    IMPLEMENTED_ALGOS,
    PolicyIteration,
    StochasticPolicyEvaluation,
    ValueIteration,
)


def get_implemented_algorithms() -> list[str]:
    """Return the implemented dynamic-programming algorithms."""
    return [algorithm.__name__ for algorithm in IMPLEMENTED_ALGOS]


get_implemented_algos = get_implemented_algorithms
get_implemented_algs = get_implemented_algorithms


__all__ = [
    "PolicyIteration",
    "StochasticPolicyEvaluation",
    "ValueIteration",
    "get_implemented_algs",
    "get_implemented_algorithms",
    "get_implemented_algos",
]
