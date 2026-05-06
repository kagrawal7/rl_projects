from .off_policy.off_policy_agent import _OffPolicyAgent


class _MCAgent:
    def __init__(self):
        self.off_policy = _OffPolicyAgent()
