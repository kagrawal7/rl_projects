
# from .gpi_algos.deterministic_gpi import DeterministicPI
# from .gpi_algos.stochastic_gpi import StochasticPI
# from .gpi_algos.value_iteration import ValueIterClass
from .off_policy.off_policy_agent import _OffPolicyAgent

class _MCAgent:
    def __init__(self):
        # self.value_iteration = ValueIterClass()
        # self.deterministic = DeterministicPI()
        # self.stochastic = StochasticPI()
        self.off_policy = _OffPolicyAgent()
