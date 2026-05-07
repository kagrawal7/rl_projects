from rl_suite.algorithms.base.temporal_difference import _TDBaseAgent


class _Sarsa(_TDBaseAgent):
    """On-policy SARSA."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )
