"""Public temporal-difference control algorithms."""

from rl_suite._algorithms._td._td_algos._expected_sarsa import _ExpectedSarsa
from rl_suite._algorithms._td._td_algos._q_learning import _QLearning
from rl_suite._algorithms._td._td_algos._sarsa import _Sarsa


class QLearning(_QLearning):
    """Tabular off-policy Q-learning control agent."""


class SARSA(_Sarsa):
    """Tabular on-policy SARSA control agent."""


class ExpectedSARSA(_ExpectedSarsa):
    """Tabular Expected SARSA control agent."""


IMPLEMENTED_ALGOS = (QLearning, SARSA, ExpectedSARSA)


__all__ = ["ExpectedSARSA", "IMPLEMENTED_ALGOS", "QLearning", "SARSA"]
