from ._gpi._gpi_agent import _GPI
from ._mc._mc_agent import _MC
from ._td._td_agent import _TD


class Algorithms:
    """Main client for interacting with the AI Library API."""

    def __init__(self, env):
        # Initialize resources
        self.set_env(env)

    def _init_agents(self):
        env = self.env
        self.gpi = _GPI(env)
        self.mc = _MC(env)
        self.td = _TD(env)
    
    def set_env(self, new_env):
        self.env = new_env
        self._init_agents()
        

# from ._gpi._gpi_algos._deterministic_gpi import _Deterministic
# from ._gpi._gpi_algos._stochastic_gpi import _Stochastic
# from ._gpi._gpi_algos._value_iteration import _ValueIteration
# from ._mc._off_policy._off_policy_control import _OffPolicyAgent
# from ._td._td_algos._expected_sarsa import _ExpectedSarsa
# from ._td._td_algos._q_learning import _QLearning
# from ._td._td_algos._sarsa import _Sarsa


# class _GPIAlgorithms:
#     def deterministic(self, *args, **kwargs):
#         return _Deterministic(*args, **kwargs)

#     def stochastic(self, *args, **kwargs):
#         return _Stochastic(*args, **kwargs)

#     def value_iteration(self, *args, **kwargs):
#         return _ValueIteration(*args, **kwargs)


# class _MCAlgorithms:
#     def off_policy(self, *args, **kwargs):
#         return _OffPolicyAgent(*args, **kwargs)


# class _TDAlgorithms:
#     def sarsa(self, *args, **kwargs):
#         return _Sarsa(*args, **kwargs)

#     def q_learning(self, *args, **kwargs):
#         return _QLearning(*args, **kwargs)

#     def expected_sarsa(self, *args, **kwargs):
#         return _ExpectedSarsa(*args, **kwargs)


# class Algorithms:
#     """Main client for interacting with the AI Library API."""

#     def __init__(self):
#         self.gpi = _GPIAlgorithms()
#         self.mc = _MCAlgorithms()
#         self.td = _TDAlgorithms()
