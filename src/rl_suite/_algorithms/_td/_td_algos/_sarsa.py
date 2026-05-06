from ._td_base import _TDAgent
import numpy as np

class _Sarsa(_TDAgent):
    """On-policy SARSA."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )
