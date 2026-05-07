from __future__ import annotations

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from rl_suite._utilities.environment import get_discrete_state, get_state_shape


class Agent(ABC):
    """Base class for agents that interact with a Gymnasium environment."""

    def __init__(self, env):
        super().__init__()
        self.env = env

    def get_env_info(self, env):
        if hasattr(env, "get_spaces"):
            states, actions, num_states, num_actions = env.get_spaces()
        else:
            num_states = env.observation_space.n
            num_actions = env.action_space.n
            states = list(range(num_states))
            actions = list(range(num_actions))
        self.S_plus = states
        self.A = actions
        self.n = num_states
        self.k = num_actions

    @abstractmethod
    def select_action(self, state):
        pass


class DynamicProgrammingAgent(Agent):
    """Base class for finite-MDP dynamic-programming algorithms."""

    def __init__(self, env, gamma, theta):
        super().__init__(env)
        self.gamma = gamma
        self.theta = theta
        self.no_policy_set = True

    def _expected_return(self, action_vector):
        """Return expected value for one transition tuple."""
        s_prime, reward = action_vector[1:3]
        return reward + self.gamma * self.V[s_prime]

    def _action_argmax(self, state):
        """Return the action that maximizes expected return."""
        transitions = self.p[state]
        return max(
            transitions,
            key=lambda action: self._expected_return(transitions[action][0]),
        )

    def get_env_info(self, env):
        super().get_env_info(env)
        self.p = env.unwrapped.P
        self.V = []
        self.terminals = set()
        for state in range(self.n):
            for action_key in self.p[state]:
                transition = self.p[state][action_key][0]
                if transition[-1]:
                    self.V.append(0)
                    self.terminals.add(transition[1])
                else:
                    self.V.append(1)
        if self.no_policy_set:
            self._initialize_policy()

    def policy_evaluation(self, print_num_iter=False, callbacks=None):
        """Evaluate the current policy with iterative policy evaluation."""
        if self.theta <= 0:
            raise ValueError("Theta must be positive number!")
        num_iter = 0
        while True:
            delta = 0
            num_iter += 1
            for state in range(self.n):
                if state in self.terminals:
                    continue
                old_value = self.V[state]
                self.V[state] = self._value_update(state)
                delta = max(delta, abs(old_value - self.V[state]))
            if callbacks is not None:
                callbacks.on_update({
                    "phase": "policy_evaluation",
                    "sweep": num_iter,
                    "delta": delta,
                    "values": self.V,
                    "policy": self.policy,
                })
            if delta < self.theta:
                break
        self.last_evaluation_sweeps = num_iter
        if print_num_iter:
            print(
                f"theta={self.theta} and gamma={self.gamma} ====> "
                f"number of steps in evalution: {num_iter}"
            )
        return self.policy, self.V

    @abstractmethod
    def _initialize_policy(self):
        pass

    @abstractmethod
    def _value_update(self, state):
        pass


class TemporalDifferenceAgent(Agent):
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
                    state,
                    action,
                    reward,
                    next_state,
                    next_action,
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
        state_shape = get_state_shape(env.observation_space)
        self._num_actions = env.action_space.n
        self.Q = np.random.random_sample((*state_shape, self._num_actions))
        self._set_terminal_states_to_zero()

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


__all__ = ["Agent", "DynamicProgrammingAgent", "TemporalDifferenceAgent"]
