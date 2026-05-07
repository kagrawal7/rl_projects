from rl_suite._algorithms._base._td_base import _TDBaseAgent


class _Sarsa(_TDBaseAgent):
    """On-policy SARSA."""

    def update_rule(self, state, action, reward, next_state, next_action):
        return (
            reward
            + self.gamma * self.Q[next_state][next_action]
            - self.Q[state][action]
        )
