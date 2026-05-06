from .off_policy_algos.base import OffPolicyMCAgent


class _OffPolicyAgent:
    def __init__(self):
        self.control = OffPolicyMCAgent
