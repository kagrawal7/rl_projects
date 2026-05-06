# from .off_policy.off_policy_agent import _OffPolicyAgent
from .td_algos.expected_sarsa import ExpectedSarsaAgent
from .td_algos.q_learning import QLearningAgent
from .td_algos.sarsa import SarsaAgent


class _TDAgent:
    def __init__(self):
        # self.off_policy = _OffPolicyAgent()
        self.sarsa = SarsaAgent()
        self.expected_sara = ExpectedSarsaAgent()
        self.q_learning = QLearningAgent()
