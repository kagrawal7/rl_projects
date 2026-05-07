import numpy as np

from rl_suite.algorithms._base.temporal_difference import _TDBaseAgent


class ExpectedSARSA(_TDBaseAgent):
    """Tabular Expected SARSA control."""

    def __init__(self, env, **kwargs):
        if "behavior_epsilon" not in kwargs:
            kwargs["behavior_epsilon"] = kwargs.get("epsilon", 0.15) * 2
        super().__init__(env, **kwargs)

    def update_rule(self, state, action, reward, next_state, next_action):
        probabilities = self._epsilon_greedy_probabilities(next_state, self.epsilon)
        expected_value = np.dot(probabilities, self.Q[next_state])
        return reward + self.gamma * expected_value - self.Q[state][action]
