from rl_suite._algorithms.agent_base import AbstractAgent
from ._off_policy._off_policy_control import _OffPolicyAgent


class _MC:
    def __init__(self, env):
        self.off_policy = _OffPolicyAgent(env)
