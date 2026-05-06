
from rl_suite._algorithms.agent_base import AbstractAgent
from ._gpi_algos._deterministic_gpi import _Deterministic
from ._gpi_algos._stochastic_gpi import _Stochastic
from ._gpi_algos._value_iteration import _ValueIteration

class _GPI:
    def __init__(self, env):
        self.value_iteration = _ValueIteration(env)
        self.deterministic = _Deterministic(env)
        self.stochastic = _Stochastic(env)
