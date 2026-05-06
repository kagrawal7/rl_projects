from ._td_base import _TDBaseAgent
import numpy as np

class _ExpectedSarsa(_TDBaseAgent):
    """Expected SARSA with a possibly more exploratory behaviour policy."""

    def __init__(self, **kwargs):
        if "behavior_epsilon" not in kwargs:
            kwargs["behavior_epsilon"] = kwargs.get("epsilon", 0.15) * 2
        super().__init__(**kwargs)

    def update_rule(self, state, action, reward, next_state, next_action):
        probabilities = self._epsilon_greedy_probabilities(next_state, self.epsilon)
        expected_value = np.dot(probabilities, self.Q[next_state])
        return reward + self.gamma * expected_value - self.Q[state][action]
