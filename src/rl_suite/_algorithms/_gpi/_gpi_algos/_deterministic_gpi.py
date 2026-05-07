from rl_suite._algorithms._base._gpi_base import _GPIBaseAgent
import numpy as np

class _Deterministic(_GPIBaseAgent):
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

    def policy_iteration(self, print_num_iter=False, callbacks=None):
        """Policy iteration as in Sutton & Barto Ch. 4."""
        if callbacks is not None:
            callbacks.on_train_begin({
                "algorithm": "policy_iteration",
                "gamma": self.gamma,
                "theta": self.theta,
            })
        policy_stable, counter = False, 0
        evaluation_sweeps = 0
        while not policy_stable:
            policy_stable = True
            counter += 1
            self.policy_evaluation(print_num_iter=print_num_iter, callbacks=callbacks)
            evaluation_sweeps += self.last_evaluation_sweeps
            changed = 0
            for s in range(self.n):
                if s in self.terminals:
                    continue
                old_policy_action = self.policy[s]
                self.policy[s] = self._action_argmax(s)
                if self.policy[s] != old_policy_action:
                    policy_stable = False
                    changed += 1
            if callbacks is not None:
                callbacks.on_update({
                    "phase": "policy_improvement",
                    "iteration": counter,
                    "changed_actions": changed,
                    "sweep": evaluation_sweeps,
                    "values": self.V,
                    "policy": self.policy,
                })
        if print_num_iter:
            print(f"Number of iterations for policy iteration: {counter}")
        self.last_iterations = counter
        self.last_evaluation_sweeps = evaluation_sweeps
        if callbacks is not None:
            callbacks.on_train_end({
                "algorithm": "policy_iteration",
                "iterations": counter,
                "sweeps": evaluation_sweeps,
                "values": self.V,
                "policy": self.policy,
            })
        return self.V, self.policy
