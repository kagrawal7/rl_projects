"""Temporal-difference control algorithms."""

import numpy as np

from rl_suite._base import TemporalDifferenceAgent


class QLearning(TemporalDifferenceAgent):
    """Tabular off-policy Q-learning control."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action]


class SARSA(TemporalDifferenceAgent):
    """Tabular on-policy SARSA control."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )


class ExpectedSARSA(TemporalDifferenceAgent):
    """Tabular Expected SARSA control."""

    def __init__(self, env, **kwargs):
        if "behavior_epsilon" not in kwargs:
            kwargs["behavior_epsilon"] = kwargs.get("epsilon", 0.15) * 2
        super().__init__(env, **kwargs)

    def update_rule(self, state, action, reward, next_state, next_action):
        probabilities = self._epsilon_greedy_probabilities(next_state, self.epsilon)
        expected_value = np.dot(probabilities, self.Q[next_state])
        return reward + self.gamma * expected_value - self.Q[state][action]


IMPLEMENTED_ALGORITHMS = (QLearning, SARSA, ExpectedSARSA)


def get_implemented_algorithms() -> list[str]:
    """Return the implemented temporal-difference algorithms."""
    return [algorithm.__name__ for algorithm in IMPLEMENTED_ALGORITHMS]


get_implemented_algs = get_implemented_algorithms


__all__ = [
    "ExpectedSARSA",
    "QLearning",
    "SARSA",
    "get_implemented_algs",
    "get_implemented_algorithms",
]
