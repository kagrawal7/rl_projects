from .td_base import TDAgent
import numpy as np

class SarsaAgent(TDAgent):
    """On-policy SARSA."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )
