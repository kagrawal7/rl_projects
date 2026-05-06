# from .off_policy.off_policy_agent import _OffPolicyAgent
from ._td_algos._expected_sarsa import _ExpectedSarsa
from ._td_algos._q_learning import _QLearning
from ._td_algos._sarsa import _Sarsa


class _TDAgent:
    def __init__(self):
        # self.off_policy = _OffPolicyAgent()
        self.sarsa = _Sarsa()
        self.expected_sara = _QLearning()
        self.q_learning = _ExpectedSarsa()
