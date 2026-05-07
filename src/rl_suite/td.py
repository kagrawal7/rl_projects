"""Temporal-difference algorithm family facade."""

from rl_suite.algos._temporal_difference import (
    IMPLEMENTED_ALGOS,
    ExpectedSARSA,
    QLearning,
    SARSA,
)


def get_implemented_algorithms() -> list[str]:
    """Return the implemented temporal-difference algorithms."""
    return [algorithm.__name__ for algorithm in IMPLEMENTED_ALGOS]


get_implemented_algos = get_implemented_algorithms
get_implemented_algs = get_implemented_algorithms


__all__ = [
    "ExpectedSARSA",
    "QLearning",
    "SARSA",
    "get_implemented_algs",
    "get_implemented_algorithms",
    "get_implemented_algos",
]
