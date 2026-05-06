"""CartPole off-policy Monte Carlo control agents from ``cartpole/part1.ipynb``."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rl_suite.utils.cartpole import discretize_interval, even_bin_count, run_cartpole_episode


class AbstractAgent(ABC):
    """Abstract MC agent: policy without environment discretization (subclasses implement)."""

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
        def execute_environment(select_action, behaviour=None):
            return run_cartpole_episode(
                env, select_action, behaviour, close_env=False
            )

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
                cont_state, A, _ = episode[t]
                S = self.get_discrete(cont_state)
                Q, C = self.table[S][A]
                C += W
                Q = Q + (W / C) * (G - Q)
                self.table[S][A] = [Q, C]

                optimal_action = np.argmax(self.table[S][..., 0])
                self.pi[S] = optimal_action
                if optimal_action != A:
                    continue
                W *= 1 / b(S, A)

        if success:
            print(
                f"Success! The agent was able to balance the pole for at least "
                f"{num_timesteps_goal} timesteps."
            )
        else:
            print(f"Failure to meet goal after {self.MAX_ITERATIONS} iterations.")
        if close_env:
            env.close()
        return output_logs


class Agent1(AbstractAgent):
    """Off-policy MC with 4D discretization (position, velocity, angle, angular velocity)."""

    def __init__(
        self,
        gamma=0.9,
        num_bins=10,
        vel_range=(-2.5, 2.5),
        angular_vel_range=(-3.5, 3.5),
    ):
        super().__init__(gamma)
        self.discretize_spaces(num_bins, vel_range, angular_vel_range)

    def _fix_num_bins(self, num_bins):
        return even_bin_count(num_bins)

    def _discretize(self, interval):
        return discretize_interval(self.n, interval)

    def discretize_spaces(self, num_bins, vel_range, angular_vel_range):
        self.n = self._fix_num_bins(num_bins)
        self.table_dims = (self.n, self.n, self.n, self.n, 2, 2)
        pos_range = (-2.4, 2.4)
        angle_range = (-0.2095, 0.2095)
        intervals = [pos_range, vel_range, angle_range, angular_vel_range]
        self.discrete_space = [self._discretize(x) for x in intervals]

    def get_discrete(self, s):
        def mapper(i):
            bin_index = np.digitize(s[i], self.discrete_space[i]) - 1
            return min(bin_index, self.n - 1)

        return tuple(map(mapper, range(len(s))))


class Agent2(AbstractAgent):
    """Off-policy MC with 2D state (cart position and pole angle only)."""

    def __init__(self, gamma=0.9, num_bins=100):
        super().__init__(gamma)
        self.discretize_spaces(num_bins)

    def _fix_num_bins(self, num_bins):
        return even_bin_count(num_bins)

    def _discretize(self, interval):
        return discretize_interval(self.n, interval)

    def discretize_spaces(self, num_bins):
        self.n = self._fix_num_bins(num_bins)
        self.table_dims = (self.n, self.n, 2, 2)
        pos_range = (-2.4, 2.4)
        angle_range = (-0.2095, 0.2095)
        intervals = [pos_range, angle_range]
        self.discrete_space = [self._discretize(x) for x in intervals]

    def get_discrete(self, state):
        s = (state[0], state[2]) if len(state) > 2 else state

        def mapper(i):
            bin_index = np.digitize(s[i], self.discrete_space[i]) - 1
            return min(bin_index, self.n - 1)

        return tuple(map(mapper, range(len(s))))


class Agent3(Agent2):
    """Like Agent 2 with epsilon-soft behaviour and per-action importance weights."""

    def __init__(self, gamma=0.3, epsilon=0.1, num_bins=100):
        super().__init__(gamma, num_bins)
        self.epsilon = epsilon

    def behavioral(self, state=None, action=None):
        s = self.get_discrete(state)
        greedy = self.pi[s]
        non_greedy = greedy ^ 1
        weights = (1 - self.epsilon, self.epsilon)
        action = np.random.choice((greedy, non_greedy), p=weights)
        return (action, weights[int(action == non_greedy)])

    def control(
        self,
        env,
        *,
        num_timesteps_goal: int = 10000,
        close_env: bool = True,
    ):
        def execute_environment(select_action, behaviour=None):
            return run_cartpole_episode(
                env, select_action, behaviour, close_env=False
            )

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
                cont_state, A, prob, _ = episode[t]
                S = self.get_discrete(cont_state)
                Q, C = self.table[S][A]
                C += W
                Q = Q + ((W / C) * (G - Q))
                self.table[S][A] = [Q, C]

                optimal_action = np.argmax(self.table[S][..., 0])
                self.pi[S] = optimal_action
                if optimal_action != A:
                    continue
                W *= 1 / prob

        if success:
            print(
                f"Success! The agent was able to balance the pole for at least "
                f"{num_timesteps_goal} timesteps."
            )
        else:
            print(f"Failure to meet goal after {self.MAX_ITERATIONS} iterations.")
        if close_env:
            env.close()
        return output_logs
