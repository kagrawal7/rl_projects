from .td_base import TDAgent
import numpy as np

class QLearningAgent(TDAgent):
    """Off-policy Q-learning."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
