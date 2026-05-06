from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from rl_suite.utils.environment import RLEnvironmentRunner


class _OffPolicyAgent:
    """Weighted importance-sampling MC control over a discretized environment."""

    def __init__(
        self,
        *,
        gamma: float = 0.9,
        behavior: str | Callable = "uniform",
        epsilon: float = 0.1,
        max_iterations: int = 100000,
    ):
        self.gamma = gamma
        self.behavior = behavior
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.table = None
        self.pi = None
        self.output_logs: list[str] = []

    def select_action(self, state, behaviour=None):
        if behaviour is not None:
            return behaviour(state)
        return int(self.pi[state])

    def control(
        self,
        env: gym.Env | RLEnvironmentRunner,
        *,
        num_timesteps_goal: int = 10000,
        close_env: bool = True,
        test_interval: int = 1000,
    ) -> list[str]:
        runner = RLEnvironmentRunner.from_env(env)
        self._initialize_policy(runner)

        self.output_logs = []
        success = False
        for num_iter in range(1, self.max_iterations + 1):
            if test_interval and not num_iter % test_interval:
                episode = self._run_episode(runner)
                self.output_logs.append(
                    f"Iteration {num_iter}, Test {num_iter // test_interval}: "
                    f"target policy episode length {len(episode)}"
                )
                if len(episode) == num_timesteps_goal:
                    success = True
                    break

            episode = self._run_episode(runner, self.behavioral)
            self._learn_from_episode(episode)

        if success:
            print(
                f"Success! The agent was able to balance the pole for at least "
                f"{num_timesteps_goal} timesteps."
            )
        else:
            print(f"Failure to meet goal after {self.max_iterations} iterations.")
        if close_env:
            runner.close()
        return self.output_logs

    def greedy_update(self):
        self.pi = np.argmax(self.table[..., 0], axis=-1)

    def behavioral(self, state=None, action=None):
        if callable(self.behavior):
            return self.behavior(state, action)
        if self.behavior == "uniform":
            return self._uniform_behavior(state, action)
        if self.behavior == "epsilon_soft":
            return self._epsilon_soft_behavior(state, action)
        raise ValueError(f"Unsupported behavior policy: {self.behavior!r}")

    def _initialize_policy(self, runner: RLEnvironmentRunner) -> None:
        state_shape = self._state_shape(runner)
        num_actions = runner.action_space.n
        self.table = np.random.random_sample((*state_shape, num_actions, 2))
        self.table[..., 1] = 0
        self.greedy_update()

    def _state_shape(self, runner: RLEnvironmentRunner) -> tuple[int, ...]:
        observation_space = runner.observation_space
        if hasattr(observation_space, "n"):
            return (observation_space.n,)
        if hasattr(observation_space, "nvec"):
            return tuple(int(x) for x in np.asarray(observation_space.nvec).flat)
        if not runner.discrete_space:
            runner.discretize_spaces()
        return tuple(len(space) for space in runner.discrete_space)

    def _run_episode(
        self,
        runner: RLEnvironmentRunner,
        behaviour: Callable | None = None,
    ) -> list[tuple]:
        return runner.run_episode(
            self.select_action,
            behaviour,
            discretize_actions=True,
            store_initial_state=True,
        )

    def _learn_from_episode(self, episode: list[tuple]) -> None:
        returns, weight = 0, 1
        for t in range(len(episode) - 2, -1, -1):
            returns = returns * self.gamma + episode[t + 1][-1]
            state, action, *rest = episode[t]
            probability = rest[0] if len(rest) == 2 else self.behavioral(state, action)
            q_value, c_value = self.table[state][action]
            c_value += weight
            q_value = q_value + (weight / c_value) * (returns - q_value)
            self.table[state][action] = [q_value, c_value]

            optimal_action = int(np.argmax(self.table[state][..., 0]))
            self.pi[state] = optimal_action
            if optimal_action != action:
                continue
            weight *= 1 / probability

    def _uniform_behavior(self, state=None, action=None):
        num_actions = self.table.shape[-2]
        if state is not None and action is not None:
            return 1 / num_actions
        return int(np.random.choice(num_actions))

    def _epsilon_soft_behavior(self, state=None, action=None):
        greedy = int(self.pi[state])
        num_actions = self.table.shape[-2]
        probabilities = np.full(num_actions, self.epsilon / (num_actions - 1))
        probabilities[greedy] = 1 - self.epsilon
        if action is not None:
            return float(probabilities[action])
        action = int(np.random.choice(num_actions, p=probabilities))
        return action, float(probabilities[action])


