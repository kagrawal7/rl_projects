from ._gpi_base import _GPIBase
import numpy as np

class _Stochastic(_GPIBase):
    """Agent that uses a stochastic policy function."""

    def __init__(self, env, gamma=0.9, theta=0.02):
        super().__init__(env, gamma, theta)

    def _intialize_policy(self):
        self.policy = np.full((self.n, self.k), 1 / self.k)
        self.no_policy_set = False

    def _value_update(self, s):
        total = 0
        action_vecs = self.p[s]
        for action_key in action_vecs:
            vec = action_vecs[action_key][0]
            total += self.policy[s][action_key] * self._expected_return(vec)
        return total

    def select_action(self, state):
        weights = self.policy[state]
        return np.random.choice(self.A, p=weights)