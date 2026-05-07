import numpy as np

from rl_suite.algorithms._base.temporal_difference import _TDBaseAgent


class QLearning(_TDBaseAgent):
    """Tabular off-policy Q-learning control."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]
