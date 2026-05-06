
from ._gpi_algos._deterministic_gpi import _Deterministic
from ._gpi_algos._stochastic_gpi import _Stochastic
from ._gpi_algos._value_iteration import _ValueIteration

class _GPIAgent:
    def __init__(self):
        self.value_iteration = _ValueIteration()
        self.deterministic = _Deterministic()
        self.stochastic = _Stochastic()
