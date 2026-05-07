"""Public reinforcement-learning algorithm classes.

The modules under :mod:`rl_suite.algos` provide the direct import path for
users who already know which algorithm they want.
"""

from ._dynamic_programming import (
    PolicyIteration,
    StochasticPolicyEvaluation,
    ValueIteration,
)
from ._monte_carlo import OffPolicyMC, OffPolicyMonteCarlo
from ._temporal_difference import ExpectedSARSA, QLearning, SARSA


IMPLEMENTED_ALGOS = (
    PolicyIteration,
    StochasticPolicyEvaluation,
    ValueIteration,
    OffPolicyMonteCarlo,
    QLearning,
    SARSA,
    ExpectedSARSA,
)


def get_implemented_algorithms() -> list[str]:
    """Return the public algorithm classes exposed by this package."""
    return [algorithm.__name__ for algorithm in IMPLEMENTED_ALGOS]


get_implemented_algos = get_implemented_algorithms
get_implemented_algs = get_implemented_algorithms


__all__ = [
    "ExpectedSARSA",
    "OffPolicyMC",
    "OffPolicyMonteCarlo",
    "PolicyIteration",
    "QLearning",
    "SARSA",
    "StochasticPolicyEvaluation",
    "ValueIteration",
    "get_implemented_algs",
    "get_implemented_algorithms",
    "get_implemented_algos",
]
