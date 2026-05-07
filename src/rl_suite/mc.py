"""Monte Carlo algorithm family facade."""

from rl_suite.algos._monte_carlo import (
    IMPLEMENTED_ALGOS,
    OffPolicyMC,
    OffPolicyMonteCarlo,
)


def get_implemented_algorithms() -> list[str]:
    """Return the implemented Monte Carlo algorithms."""
    return [algorithm.__name__ for algorithm in IMPLEMENTED_ALGOS]


get_implemented_algos = get_implemented_algorithms
get_implemented_algs = get_implemented_algorithms


__all__ = [
    "OffPolicyMC",
    "OffPolicyMonteCarlo",
    "get_implemented_algs",
    "get_implemented_algorithms",
    "get_implemented_algos",
]
