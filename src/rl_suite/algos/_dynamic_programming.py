"""Public dynamic-programming algorithms."""

from rl_suite._algorithms._gpi._gpi_algos._deterministic_gpi import _Deterministic
from rl_suite._algorithms._gpi._gpi_algos._stochastic_gpi import _Stochastic
from rl_suite._algorithms._gpi._gpi_algos._value_iteration import _ValueIteration


class PolicyIteration(_Deterministic):
    """Dynamic-programming policy iteration with a deterministic policy."""


class StochasticPolicyEvaluation(_Stochastic):
    """Dynamic-programming policy evaluation with a stochastic policy."""


class ValueIteration(_ValueIteration):
    """Dynamic-programming value iteration for finite Markov decision processes."""


IMPLEMENTED_ALGOS = (
    PolicyIteration,
    StochasticPolicyEvaluation,
    ValueIteration,
)


__all__ = [
    "IMPLEMENTED_ALGOS",
    "PolicyIteration",
    "StochasticPolicyEvaluation",
    "ValueIteration",
]
