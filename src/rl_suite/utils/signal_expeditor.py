from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


def even_bin_count(n: int, min_val: int = 6) -> int:
    """Return an even integer bin count at least ``min_val``."""
    n = int(n)
    if n < min_val:
        raise ValueError(f"Cannot specify number smaller than {min_val}")
    return n + 1 if n % 2 else n


def discretize_interval(n: int, interval: tuple[float, float]) -> np.ndarray:
    """Create symmetric bins around 0 for one scalar observation interval."""
    lower_bound, upper_bound = interval
    mid = n // 2
    lower = np.linspace(lower_bound, 0, mid, endpoint=False)
    upper = np.linspace(0, upper_bound, mid + 1)
    return np.concatenate([lower, upper])


@dataclass(frozen=True)
class DiscretizationConfig:
    """How to turn Box observations into tabular state indices."""

    num_bins: int = 10
    intervals: Sequence[tuple[float, float]] | None = None
    feature_indices: Sequence[int] | None = None
    infinite_bound_interval: tuple[float, float] = (-1.0, 1.0)
    min_bins: int = 6
    include_terminal_bin: bool = False
    symmetric_bins: bool = True


class SignalExpeditor:
    """Small Gymnasium adapter used by the tabular algorithms."""

    def __init__(
        self,
        env: gym.Env,
        discretization: DiscretizationConfig | dict | None = None,
    ):
        self.env = env
        self.discretization = self._config(discretization)
        self.discrete_space: list[np.ndarray] = []
        self.n: int | None = None
        self.n_plus: int | None = None
        if self.discretization is not None:
            self.discretize_spaces()

    @classmethod
    def from_env(
        cls,
        env_or_expeditor: gym.Env | "SignalExpeditor",
        discretization: DiscretizationConfig | dict | None = None,
    ) -> "SignalExpeditor":
        if isinstance(env_or_expeditor, cls):
            if discretization is not None:
                env_or_expeditor.discretization = cls._config(discretization)
                env_or_expeditor.discretize_spaces()
            return env_or_expeditor
        return cls(env_or_expeditor, discretization=discretization)

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def get_spaces(self):
        """Return ``S, A, n, k`` for discrete environments."""
        if not hasattr(self.observation_space, "n") or not hasattr(
            self.action_space, "n"
        ):
            raise TypeError("get_spaces requires discrete observation and action spaces")
        n, k = self.observation_space.n, self.action_space.n
        return list(range(n)), list(range(k)), n, k

    def discretize_spaces(
        self,
        num_bins: int | None = None,
        intervals: Sequence[tuple[float, float]] | None = None,
        feature_indices: Sequence[int] | None = None,
        include_terminal_bin: bool | None = None,
    ) -> list[np.ndarray]:
        """Build per-feature bin arrays for a Box observation space."""
        config = self.discretization or DiscretizationConfig()
        num_bins = config.num_bins if num_bins is None else int(num_bins)
        feature_indices = feature_indices or config.feature_indices
        intervals = intervals or config.intervals
        include_terminal_bin = (
            config.include_terminal_bin
            if include_terminal_bin is None
            else include_terminal_bin
        )
        self.n = even_bin_count(num_bins, min_val=config.min_bins)
        self.n_plus = self.n + 1 if include_terminal_bin else self.n

        if not isinstance(self.observation_space, gym.spaces.Box):
            self.discrete_space = []
            return self.discrete_space

        low = np.asarray(self.observation_space.low, dtype=float).reshape(-1)
        high = np.asarray(self.observation_space.high, dtype=float).reshape(-1)
        indices = tuple(feature_indices or range(len(low)))

        if intervals is None:
            intervals = []
            for index in indices:
                lower, upper = low[index], high[index]
                if not np.isfinite(lower) or not np.isfinite(upper):
                    lower, upper = config.infinite_bound_interval
                intervals.append((float(lower), float(upper)))

        if len(intervals) != len(indices):
            raise ValueError("Number of intervals must match selected feature count")

        if config.symmetric_bins:
            self.discrete_space = [discretize_interval(self.n, x) for x in intervals]
        else:
            self.discrete_space = [
                np.linspace(lo, hi, self.n + 1) for lo, hi in intervals
            ]

        self.discretization = DiscretizationConfig(
            num_bins=self.n,
            intervals=tuple(intervals),
            feature_indices=tuple(indices),
            infinite_bound_interval=config.infinite_bound_interval,
            min_bins=config.min_bins,
            include_terminal_bin=include_terminal_bin,
            symmetric_bins=config.symmetric_bins,
        )
        return self.discrete_space

    def get_discrete(self, state):
        """Convert an observation into a table-friendly discrete state."""
        if isinstance(self.observation_space, gym.spaces.Discrete):
            return int(state)
        if isinstance(self.observation_space, gym.spaces.MultiDiscrete):
            return tuple(int(x) for x in np.asarray(state).flat)
        if not isinstance(self.observation_space, gym.spaces.Box):
            return state

        if self.discretization is None or not self.discrete_space:
            self.discretize_spaces()

        values = np.asarray(state, dtype=float).reshape(-1)
        indices = self.discretization.feature_indices
        discrete_state = []
        for position, index in enumerate(indices):
            value = values[index]
            bins = self.discrete_space[position]
            if self.discretization.include_terminal_bin:
                if value == bins[-1]:
                    discrete_state.append(len(bins) - 1)
                elif value > bins[-1]:
                    discrete_state.append(0)
                else:
                    discrete_state.append(int(np.digitize(value, bins)))
            else:
                bin_index = int(np.digitize(value, bins)) - 1
                discrete_state.append(min(max(bin_index, 0), len(bins) - 1))
        return tuple(discrete_state)

    def execute_environment(
        self,
        select_action: Callable | None = None,
        behaviour: Callable | None = None,
        *,
        agent=None,
        seed: int | None = None,
        close_env: bool = False,
        discretize_actions: bool = False,
        human: bool = False,
        action_message: str | None = None,
        invalid_action_message: Callable[[str], str] | None = None,
        step_message: Callable | None = None,
        final_message: Callable | None = None,
        show_messages: bool = False,
        render_fn: Callable | None = None,
        render_each_step: bool = False,
        store_initial_state: bool = False,
    ) -> list[tuple]:
        """
        Roll out one episode and return transition tuples.

        With no behaviour policy, transitions are ``(state, action, reward)``.
        With a behaviour policy that returns ``(action, probability)``, transitions
        are ``(state, action, probability, reward)``.
        """
        def log(message: str | None) -> None:
            if show_messages and message:
                print(message)

        if select_action is None and not human:
            select_action = (
                agent.select_action if agent else lambda _: self.action_space.sample()
            )
        if agent is not None and hasattr(agent, "get_env_info"):
            agent.get_env_info(self)

        episode: list[tuple] = []
        reset_kwargs = {} if seed is None else {"seed": seed}
        state, _ = self.env.reset(**reset_kwargs)
        terminated = truncated = exited = False
        finished = False

        while not finished:
            if render_each_step and render_fn is not None:
                render_fn(self)
            elif render_each_step:
                self.env.render()

            action_state = self.get_discrete(state) if discretize_actions else state
            if human:
                log(action_message)
                user_input = input()
                if user_input.lower() == "exit":
                    exited = True
                    break
                try:
                    action_output = int(user_input)
                except ValueError:
                    if invalid_action_message is not None:
                        log(invalid_action_message(user_input))
                    continue
            elif behaviour is None:
                action_output = select_action(action_state)
            else:
                action_output = select_action(action_state, behaviour)

            if isinstance(action_output, tuple) and len(action_output) == 2:
                action, probability = action_output
            else:
                action, probability = action_output, None

            next_state, reward, terminated, truncated, info = self.env.step(action)
            stored_state = state if store_initial_state else next_state
            if probability is None:
                episode.append((stored_state, action, reward))
            else:
                episode.append((stored_state, action, probability, reward))

            if step_message is not None:
                log(step_message(action, next_state, reward, info))

            state = next_state
            finished = terminated or truncated

        if final_message is not None:
            log(final_message(state, terminated, truncated, exited))
        if close_env:
            self.env.close()
        return episode

    run_episode = execute_environment

    @staticmethod
    def _config(discretization: DiscretizationConfig | dict | None):
        if discretization is None or isinstance(discretization, DiscretizationConfig):
            return discretization
        return DiscretizationConfig(**discretization)


__all__ = [
    "DiscretizationConfig",
    "SignalExpeditor",
    "discretize_interval",
    "even_bin_count",
]
