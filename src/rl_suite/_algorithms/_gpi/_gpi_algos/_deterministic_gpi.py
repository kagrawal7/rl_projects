from ._gpi_base import _GPIBase
import numpy as np

class _Deterministic(_GPIBase):
    """Deterministic policy table with policy iteration."""

    def __init__(self, env, gamma=0.9, theta=0.02):
        super().__init__(env, gamma, theta)

    def _intialize_policy(self):
        self.policy = [np.random.choice(self.A) for s in range(self.n)]
        self.no_policy_set = False

    def _value_update(self, s):
        action = self.policy[s]
        return self._expected_return(self.p[s][action][0])

    def select_action(self, state):
        return self.policy[state]

    def policy_iteration(self, print_num_iter=False):
        """Policy iteration as in Sutton & Barto Ch. 4."""
        policy_stable, counter = False, 0
        while not policy_stable:
            policy_stable = True
            counter += 1
            self.policy_evaluation(print_num_iter=print_num_iter)
            for s in range(self.n):
                if s in self.terminals:
                    continue
                old_policy_action = self.policy[s]
                self.policy[s] = self._action_argmax(s)
                if self.policy[s] != old_policy_action:
                    policy_stable = False
        if print_num_iter:
            print(f"Number of iterations for policy iteration: {counter}")
        return self.V, self.policy
