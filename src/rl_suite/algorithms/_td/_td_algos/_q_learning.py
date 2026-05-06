from ._td_base import _TDAgent
import numpy as np

class _QLearning(_TDAgent):
    """Off-policy Q-learning."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
