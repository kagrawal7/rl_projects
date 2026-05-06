from ._off_policy._off_policy_control import _OffPolicyAgent


class _MCAgent:
    def __init__(self):
        self.off_policy = _OffPolicyAgent()
