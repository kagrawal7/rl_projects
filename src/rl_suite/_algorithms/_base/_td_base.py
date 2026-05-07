from __future__ import annotations

import gymnasium as gym
import numpy as np

from ._agent_base import AbstractAgent
from rl_suite._utilities.environment import get_discrete_state, get_state_shape


class _TDBaseAgent(AbstractAgent):
    """Base tabular TD control agent over a discrete Gymnasium environment."""

    def __init__(
        self,
        env,
        *,
        gamma: float = 0.9,
        alpha: float = 0.2,
        epsilon: float = 0.15,
        behavior_epsilon: float | None = None,
        max_iterations: int = 50000,
    ):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.behavior_epsilon = (
            epsilon if behavior_epsilon is None else behavior_epsilon
        )
        self.max_iterations = max_iterations
        self.Q = None
        self.output_logs: list[str] = []

    def control(
        self,
        env: gym.Env,
        *,
        num_timesteps_goal: int = 10000,
        close_env: bool = True,
        log_interval: int = 5000,
    ) -> list[str]:
        self._initialize_values(env)

        self.output_logs = []
        success = False
        for num_iter in range(1, self.max_iterations + 1):
            state, _ = env.reset()
            state = get_discrete_state(state, env.observation_space)
            action = self.select_action(state, behavior=True)
            count = 0
            finished = False

            while not finished:
                count += 1
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = get_discrete_state(next_state, env.observation_space)
                next_action = self.select_action(next_state, behavior=True)

                update = self.update_rule(
                    state, action, reward, next_state, next_action
                )
                self.Q[state][action] += self.alpha * update

                state, action = next_state, next_action
                if truncated:
                    success = True
                    break
                finished = terminated

            if log_interval and not num_iter % log_interval:
                message = (
                    f"Iteration number: {num_iter}. "
                    f"Duration of episode: {count} timesteps."
                )
                self.output_logs.append(message)
                print(f"{message} \n\n")

            if success:
                break

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

    def update_rule(self, state, action, reward, next_state, next_action):
        raise NotImplementedError

    def select_action(self, state, behavior: bool = False):
        epsilon = self.behavior_epsilon if behavior else self.epsilon
        probabilities = self._epsilon_greedy_probabilities(state, epsilon)
        return int(np.random.choice(self._num_actions, p=probabilities))

    def get_greedy_action(self, state):
        return int(np.argmax(self.Q[state]))

    def _initialize_values(self, env: gym.Env) -> None:
        state_shape = self._state_shape(env)
        self._num_actions = env.action_space.n
        self.Q = np.random.random_sample((*state_shape, self._num_actions))
        self._set_terminal_states_to_zero()

    def _state_shape(self, env: gym.Env) -> tuple[int, ...]:
        return get_state_shape(env.observation_space)

    def _set_terminal_states_to_zero(self) -> None:
        state_dimensions = self.Q.shape[:-1]
        for axis in range(len(state_dimensions)):
            index = [slice(None)] * self.Q.ndim
            index[axis] = 0
            self.Q[tuple(index)] = 0

    def _epsilon_greedy_probabilities(self, state, epsilon: float) -> np.ndarray:
        probabilities = np.full(self._num_actions, epsilon / self._num_actions)
        probabilities[self.get_greedy_action(state)] += 1 - epsilon
        return probabilities



__all__ = ["_TDBaseAgent"]
