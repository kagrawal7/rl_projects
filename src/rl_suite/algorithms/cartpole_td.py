"""CartPole temporal-difference agents from ``cartpole/part2.ipynb``."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rl_suite.utils.cartpole import discretize_interval, even_bin_count
from rl_suite.utils.signal_expeditor import SignalExpeditor


class TDAgent(ABC):
    """Base tabular TD agent with terminal-aware discretization."""

    def __init__(
        self,
        gamma=0.9,
        alpha=0.2,
        epsilon=0.15,
        num_bins=14,
        vel_range=(-2.5, 2.5),
        angular_vel_range=(-3.5, 3.5),
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.MAX_ITERATIONS = 100000
        self.discretize_spaces(num_bins, vel_range, angular_vel_range)
        self.Q = []

        self.behaviour = None

    @abstractmethod
    def update_rule(self):
        pass

    @abstractmethod
    def select_action(self, state, behaviour=False):
        pass

    def discretize_spaces(self, num_bins, vel_range, angular_vel_range):
        self.n = even_bin_count(num_bins)
        self.n_plus = self.n + 1
        self.table_dims = (self.n_plus, self.n_plus, self.n_plus, self.n_plus, 2)
        pos_range = (-2.4, 2.4)
        angle_range = (-0.2095, 0.2095)
        intervals = [pos_range, vel_range, angle_range, angular_vel_range]
        self.discrete_space = [discretize_interval(self.n, x) for x in intervals]

    def get_discrete(self, state):
        def mapper(i):
            s, bins = state[i], self.discrete_space[i]
            if s == bins[-1]:
                return self.n
            elif s > bins[-1]:
                return 0
            bin_index = np.digitize(s, bins)
            return bin_index

        return tuple(map(mapper, range(len(state))))

    def get_greedy_action(self, state):
        return np.argmax(self.Q[state])

    def set_terminals_to_zero(self):
        j = np.arange(self.n_plus)
        self.Q[0, j, j, j, :] = 0
        self.Q[j, 0, j, j, :] = 0
        self.Q[j, j, 0, j, :] = 0
        self.Q[j, j, j, 0, :] = 0

    def control(
        self,
        env,
        *,
        num_timesteps_goal: int = 10000,
        close_env: bool = True,
    ):
        expeditor = SignalExpeditor.from_env(env)
        self.Q = np.random.random_sample(self.table_dims)
        self.set_terminals_to_zero()

        output_logs = []
        success = False

        for num_iter in range(1, self.MAX_ITERATIONS + 1):
            state, _ = expeditor.reset()
            state = self.get_discrete(state)
            reward, finished = None, False

            count = 0
            while not finished:
                count += 1
                s = state
                action = self.select_action(state, self.behaviour)

                state, reward, terminated, truncated, _ = expeditor.step(action)
                state = self.get_discrete(state)

                a, r, s_prime = action, reward, state
                a_prime = self.select_action(state, self.behaviour)

                curr = self.Q[s][a]
                update = self.update_rule(s, a, r, s_prime, a_prime)
                new_val = self.alpha * (update) + curr
                self.Q[s][a] = new_val

                if truncated:
                    success = True
                    break
                if terminated:
                    finished = True

            if not num_iter % 5000:
                print(
                    f"Iteration number: {num_iter}. "
                    f"Duration of episode: {count} timesteps. \n\n"
                )

            if success:
                break

        if success:
            print(
                f"Success! The agent was able to balance the pole for at least "
                f"{num_timesteps_goal} timesteps."
            )
        else:
            print(f"Failure to meet goal after {self.MAX_ITERATIONS} iterations.")
        if close_env:
            expeditor.close()
        return output_logs


class AgentA(TDAgent):
    """On-policy SARSA-style agent (notebook naming)."""

    def __init__(
        self,
        gamma=0.9,
        alpha=0.2,
        epsilon=0.15,
        num_bins=14,
        vel_range=(-2.5, 2.5),
        angular_vel_range=(-3.5, 3.5),
    ):
        super().__init__(gamma, alpha, epsilon, num_bins, vel_range, angular_vel_range)

    def update_rule(self, s, a, r, s_prime, a_prime):
        return self.gamma * np.max(self.Q[s_prime]) - self.Q[s][a] + r

    def select_action(self, state, behaviour=False):
        greedy = self.get_greedy_action(state)
        non_greedy = greedy ^ 1
        if np.random.random_sample() <= self.epsilon:
            return non_greedy
        return greedy


class AgentB(AgentA):
    """TD update uses bootstrap on ``a_prime`` (notebook off-policy variant)."""

    def __init__(
        self,
        gamma=0.9,
        alpha=0.2,
        epsilon=0.15,
        num_bins=14,
        vel_range=(-2.5, 2.5),
        angular_vel_range=(-3.5, 3.5),
    ):
        super().__init__(gamma, alpha, epsilon, num_bins, vel_range, angular_vel_range)

    def update_rule(self, s, a, r, s_prime, a_prime):
        return self.gamma * self.Q[s_prime][a_prime] - self.Q[s][a] + r


class AgentC(TDAgent):
    """Expected SARSA-style update with separate behaviour epsilon."""

    def __init__(
        self,
        gamma=0.9,
        alpha=0.2,
        epsilon=0.15,
        num_bins=14,
        vel_range=(-2.5, 2.5),
        angular_vel_range=(-3.5, 3.5),
    ):
        super().__init__(gamma, alpha, epsilon, num_bins, vel_range, angular_vel_range)
        self.behaviour = True
        self.b_epsilon = epsilon * 2

    def select_action(self, state, behavior=False):
        greedy = self.get_greedy_action(state)
        non_greedy = greedy ^ 1
        threshold = self.b_epsilon if behavior else self.epsilon
        if np.random.random_sample() <= threshold:
            return non_greedy
        return greedy

    def update_rule(self, s, a, r, s_prime, a_prime):
        greedy_val = np.max(self.Q[s_prime])
        non_greedy_val = np.min(self.Q[s_prime])
        expected_val_est = greedy_val * (1 - self.epsilon) + non_greedy_val * self.epsilon
        return self.gamma * expected_val_est - self.Q[s][a] + r
