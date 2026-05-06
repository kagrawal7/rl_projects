from __future__ import annotations

from abc import abstractmethod

import numpy as np

from rl_suite.algorithms.agent_base import AbstractAgent
from rl_suite.utils.environment import RLEnvironmentRunner


class _MCOffPolicyBase(AbstractAgent):
    """Abstract MC agent: policy without environment discretization."""

    def __init__(self, gamma):
        self.gamma = gamma
        self.MAX_ITERATIONS = 100000

    @abstractmethod
    def discretize_spaces(self):
        pass

    @abstractmethod
    def get_discrete(self):
        pass

    def select_action(self, state, b=None):
        s = self.get_discrete(state)
        if b is not None:
            return b(s)
        return self.pi[s]

    def greedy_update(self):
        self.pi = np.argmax(self.table[..., 0], axis=-1)

    def behavioral(self, state=None, action=None):
        if state is not None and action is not None:
            return 0.5
        return np.random.choice(2, 1)[0]

    def control(
        self,
        env,
        *,
        num_timesteps_goal: int = 10000,
        close_env: bool = True,
    ):
        runner = RLEnvironmentRunner.from_env(env)

        def execute_environment(select_action, behaviour=None):
            return runner.run_episode(select_action, behaviour)

        self.table = np.random.random_sample(self.table_dims)
        self.table[..., 1] = 0
        self.greedy_update()

        output_logs = []
        success = False

        for num_iter in range(1, self.MAX_ITERATIONS + 1):
            if not num_iter % 1000:
                ep = execute_environment(self.select_action)
                output_logs.append(
                    f"Iteration {num_iter}, Test {num_iter // 1000}: "
                    f"target policy episode length {len(ep)}"
                )
                if len(ep) == num_timesteps_goal:
                    success = True
                    break

            G, W = 0, 1
            b = self.behavioral

            episode = execute_environment(self.select_action, b)
            for t in range(len(episode) - 2, -1, -1):
                G = G * self.gamma + episode[t + 1][-1]
                cont_state, action, _ = episode[t]
                state = self.get_discrete(cont_state)
                q_value, c_value = self.table[state][action]
                c_value += W
                q_value = q_value + (W / c_value) * (G - q_value)
                self.table[state][action] = [q_value, c_value]

                optimal_action = np.argmax(self.table[state][..., 0])
                self.pi[state] = optimal_action
                if optimal_action != action:
                    continue
                W *= 1 / b(state, action)

        if success:
            print(
                f"Success! The agent was able to balance the pole for at least "
                f"{num_timesteps_goal} timesteps."
            )
        else:
            print(f"Failure to meet goal after {self.MAX_ITERATIONS} iterations.")
        if close_env:
            runner.close()
        return output_logs


__all__ = ["_MCOffPolicyBase"]
