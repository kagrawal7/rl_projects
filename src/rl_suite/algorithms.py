"""Flat access to all implemented reinforcement-learning algorithms."""

from rl_suite.dp import PolicyIteration, StochasticPolicyEvaluation, ValueIteration
from rl_suite.mc import OffPolicyMC, OffPolicyMonteCarlo
from rl_suite.td import ExpectedSARSA, QLearning, SARSA


IMPLEMENTED_ALGORITHMS = (
    PolicyIteration,
    StochasticPolicyEvaluation,
    ValueIteration,
    OffPolicyMonteCarlo,
    QLearning,
    SARSA,
    ExpectedSARSA,
)


def get_implemented_algorithms() -> list[str]:
    """Return all public algorithm classes exposed by rl_suite."""
    return [algorithm.__name__ for algorithm in IMPLEMENTED_ALGORITHMS]


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
]
