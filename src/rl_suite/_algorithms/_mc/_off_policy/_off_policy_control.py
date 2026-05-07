from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np

from rl_suite._algorithms._base._agent_base import AbstractAgent
from rl_suite._utilities.environment import get_discrete_state, get_state_shape


class _OffPolicyAgent(AbstractAgent):
    """Weighted importance-sampling MC control over a discrete environment."""

    def __init__(
        self,
        env,
        *,
        gamma: float = 0.9,
        behavior: str | Callable = "uniform",
        epsilon: float = 0.1,
        max_iterations: int = 100000,
    ):
        super().__init__(env)
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
        env: gym.Env,
        *,
        num_timesteps_goal: int = 10000,
        close_env: bool = True,
        test_interval: int = 1000,
    ) -> list[str]:
        self._initialize_policy(env)

        self.output_logs = []
        success = False
        for num_iter in range(1, self.max_iterations + 1):
            if test_interval and not num_iter % test_interval:
                episode = self._run_episode(env)
                self.output_logs.append(
                    f"Iteration {num_iter}, Test {num_iter // test_interval}: "
                    f"target policy episode length {len(episode)}"
                )
                if len(episode) == num_timesteps_goal:
                    success = True
                    break

            episode = self._run_episode(env, self.behavioral)
            self._learn_from_episode(episode)

        if success:
            print(
                f"Success! The agent was able to balance the pole for at least "
                f"{num_timesteps_goal} timesteps."
            )
        else:
            print(f"Failure to meet goal after {self.max_iterations} iterations.")
        if close_env:
            env.close()
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

    def _initialize_policy(self, env: gym.Env) -> None:
        state_shape = self._state_shape(env)
        num_actions = env.action_space.n
        self.table = np.random.random_sample((*state_shape, num_actions, 2))
        self.table[..., 1] = 0
        self.greedy_update()

    def _state_shape(self, env: gym.Env) -> tuple[int, ...]:
        return get_state_shape(env.observation_space)

    def _run_episode(
        self,
        env: gym.Env,
        behaviour: Callable | None = None,
    ) -> list[tuple]:
        episode: list[tuple] = []
        state, _ = env.reset()
        state = get_discrete_state(state, env.observation_space)
        terminated = truncated = False

        while not (terminated or truncated):
            if behaviour is None:
                action_output = self.select_action(state)
            else:
                action_output = self.select_action(state, behaviour)
            action, probability = self._normalize_action_output(action_output)

            next_state, reward, terminated, truncated, _ = env.step(action)
            if probability is None:
                episode.append((state, action, reward))
            else:
                episode.append((state, action, probability, reward))
            if not (terminated or truncated):
                state = get_discrete_state(next_state, env.observation_space)

        return episode

    @staticmethod
    def _normalize_action_output(action_output):
        if isinstance(action_output, tuple) and len(action_output) == 2:
            return action_output
        return action_output, None

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
