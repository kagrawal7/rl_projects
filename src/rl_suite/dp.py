"""Dynamic-programming algorithms."""

import numpy as np

from rl_suite._base import DynamicProgrammingAgent


class StochasticPolicyEvaluation(DynamicProgrammingAgent):
    """Policy evaluation with a stochastic policy table."""

    def __init__(self, env, gamma=0.9, theta=0.02):
        super().__init__(env, gamma, theta)

    def _initialize_policy(self):
        self.policy = np.full((self.n, self.k), 1 / self.k)
        self.no_policy_set = False

    def _value_update(self, state):
        total = 0
        transitions = self.p[state]
        for action_key in transitions:
            transition = transitions[action_key][0]
            total += self.policy[state][action_key] * self._expected_return(
                transition
            )
        return total

    def select_action(self, state):
        weights = self.policy[state]
        return np.random.choice(self.A, p=weights)


class PolicyIteration(DynamicProgrammingAgent):
    """Policy iteration with a deterministic policy table."""

    def __init__(self, env, gamma=0.9, theta=0.02):
        super().__init__(env, gamma, theta)

    def _initialize_policy(self):
        self.policy = [np.random.choice(self.A) for _ in range(self.n)]
        self.no_policy_set = False

    def _value_update(self, state):
        action = self.policy[state]
        return self._expected_return(self.p[state][action][0])

    def select_action(self, state):
        return self.policy[state]

    def policy_iteration(self, print_num_iter=False, callbacks=None):
        """Run policy iteration as in Sutton & Barto Chapter 4."""
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
            for state in range(self.n):
                if state in self.terminals:
                    continue
                old_policy_action = self.policy[state]
                self.policy[state] = self._action_argmax(state)
                if self.policy[state] != old_policy_action:
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


class ValueIteration(StochasticPolicyEvaluation):
    """Value iteration for finite Markov decision processes."""

    def __init__(self, env, gamma=0.9, theta=0.02):
        super().__init__(env, gamma, theta)

    def _value_update(self, state):
        transitions = self.p[state]
        return max(
            self.policy[state][action] * self._expected_return(
                transitions[action][0]
            )
            for action in range(self.k)
        )

    def select_action(self, state):
        return self.policy[state]

    def value_iteration(self, print_num_iter=False, callbacks=None):
        """Run value iteration as in Sutton & Barto Chapter 4."""
        if callbacks is not None:
            callbacks.on_train_begin({
                "algorithm": "value_iteration",
                "gamma": self.gamma,
                "theta": self.theta,
            })
        self.policy_evaluation(print_num_iter, callbacks=callbacks)
        self.policy = [self._action_argmax(state) for state in range(self.n)]
        self.last_iterations = self.last_evaluation_sweeps
        if callbacks is not None:
            callbacks.on_train_end({
                "algorithm": "value_iteration",
                "iterations": self.last_iterations,
                "sweeps": self.last_evaluation_sweeps,
                "values": self.V,
                "policy": self.policy,
            })
        return self.V, self.policy


IMPLEMENTED_ALGORITHMS = (
    PolicyIteration,
    StochasticPolicyEvaluation,
    ValueIteration,
)


def get_implemented_algorithms() -> list[str]:
    """Return the implemented dynamic-programming algorithms."""
    return [algorithm.__name__ for algorithm in IMPLEMENTED_ALGORITHMS]


get_implemented_algs = get_implemented_algorithms


__all__ = [
    "PolicyIteration",
    "StochasticPolicyEvaluation",
    "ValueIteration",
    "get_implemented_algs",
    "get_implemented_algorithms",
]
