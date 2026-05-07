from ._stochastic_gpi import _Stochastic

class _ValueIteration(_Stochastic):
    """Stochastic policy init + value iteration for an optimal deterministic policy."""

    def __init__(self, env, gamma=0.9, theta=0.02):
        super().__init__(env, gamma, theta)

    def _value_update(self, s):
        action_vecs = self.p[s]
        return max(
            map(
                lambda a: self.policy[s][a] * self._expected_return(action_vecs[a][0]),
                range(self.k),
            )
        )

    def select_action(self, state):
        return self.policy[state]

    def value_iteration(self, print_num_iter=False, callbacks=None):
        """Value iteration as in Sutton & Barto Ch. 4."""
        if callbacks is not None:
            callbacks.on_train_begin({
                "algorithm": "value_iteration",
                "gamma": self.gamma,
                "theta": self.theta,
            })
        self.policy_evaluation(print_num_iter, callbacks=callbacks)
        self.policy = list(map(lambda s: self._action_argmax(s), range(self.n)))
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
