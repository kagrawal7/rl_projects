# from .off_policy.off_policy_agent import _OffPolicyAgent
from rl_suite._algorithms.agent_base import AbstractAgent
from ._td_algos._expected_sarsa import _ExpectedSarsa
from ._td_algos._q_learning import _QLearning
from ._td_algos._sarsa import _Sarsa


class _TD:
    def __init__(self, env):
        # self.off_policy = _OffPolicyAgent()
        self.sarsa = _Sarsa(env)
        self.expected_sara = _QLearning(env)
        self.q_learning = _ExpectedSarsa(env)
