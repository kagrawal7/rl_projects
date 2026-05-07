from rl_suite.algorithms._base.temporal_difference import _TDBaseAgent


class SARSA(_TDBaseAgent):
    """Tabular on-policy SARSA control."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )
