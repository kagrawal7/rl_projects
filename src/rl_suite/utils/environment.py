from __future__ import annotations

from collections.abc import Callable, Sequence
import gymnasium as gym
from gymnasium.envs.registration import register, registry
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


DEFAULT_DISCRETIZATION = {
    "num_bins": 10,
    "intervals": None,
    "feature_indices": None,
    "infinite_bound_interval": (-1.0, 1.0),
    "min_bins": 6,
    "include_terminal_bin": False,
    "symmetric_bins": True,
}


class RLEnvironmentRunner:
    """Gymnasium adapter used by RL agents and tabular algorithms."""

    def __init__(
        self,
        env: gym.Env,
        discretization: dict | None = None,
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
        env_or_runner: gym.Env | "RLEnvironmentRunner",
        discretization: dict | None = None,
    ) -> "RLEnvironmentRunner":
        if isinstance(env_or_runner, cls):
            if discretization is not None:
                env_or_runner.discretization = cls._config(discretization)
                env_or_runner.discretize_spaces()
            return env_or_runner
        return cls(env_or_runner, discretization=discretization)

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
            raise TypeError(
                "get_spaces requires discrete observation and action spaces"
            )
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
        config = self.discretization or DEFAULT_DISCRETIZATION
        num_bins = config["num_bins"] if num_bins is None else int(num_bins)
        feature_indices = feature_indices or config["feature_indices"]
        intervals = intervals or config["intervals"]
        include_terminal_bin = (
            config["include_terminal_bin"]
            if include_terminal_bin is None
            else include_terminal_bin
        )
        self.n = even_bin_count(num_bins, min_val=config["min_bins"])
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
                    lower, upper = config["infinite_bound_interval"]
                intervals.append((float(lower), float(upper)))

        if len(intervals) != len(indices):
            raise ValueError("Number of intervals must match selected feature count")

        if config["symmetric_bins"]:
            self.discrete_space = [discretize_interval(self.n, x) for x in intervals]
        else:
            self.discrete_space = [
                np.linspace(lo, hi, self.n + 1) for lo, hi in intervals
            ]

        self.discretization = {
            "num_bins": self.n,
            "intervals": tuple(intervals),
            "feature_indices": tuple(indices),
            "infinite_bound_interval": config["infinite_bound_interval"],
            "min_bins": config["min_bins"],
            "include_terminal_bin": include_terminal_bin,
            "symmetric_bins": config["symmetric_bins"],
        }
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
        indices = self.discretization["feature_indices"]
        discrete_state = []
        for position, index in enumerate(indices):
            value = values[index]
            bins = self.discrete_space[position]
            if self.discretization["include_terminal_bin"]:
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

    def run_episode(
        self,
        select_action: Callable | None = None,
        behaviour: Callable | None = None,
        *,
        agent=None,
        seed: int | None = None,
        close_env: bool = False,
        discretize_actions: bool = False,
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
        if select_action is None:
            select_action = (
                agent.select_action if agent else lambda _: self.action_space.sample()
            )
        if agent is not None and hasattr(agent, "get_env_info"):
            agent.get_env_info(self)

        episode: list[tuple] = []
        reset_kwargs = {} if seed is None else {"seed": seed}
        state, _ = self.env.reset(**reset_kwargs)
        terminated = truncated = False

        while not (terminated or truncated):
            if render_each_step and render_fn is not None:
                render_fn(self)
            elif render_each_step:
                self.env.render()

            action_state = self.get_discrete(state) if discretize_actions else state
            if behaviour is None:
                action_output = select_action(action_state)
            else:
                action_output = select_action(action_state, behaviour)
            action, probability = self._normalize_action_output(action_output)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            if store_initial_state:
                stored_state = action_state if discretize_actions else state
            else:
                stored_state = (
                    self.get_discrete(next_state) if discretize_actions else next_state
                )
            if probability is None:
                episode.append((stored_state, action, reward))
            else:
                episode.append((stored_state, action, probability, reward))
            state = next_state

        if close_env:
            self.env.close()
        return episode

    execute_environment = run_episode
    run = run_episode

    @staticmethod
    def _normalize_action_output(action_output):
        if isinstance(action_output, tuple) and len(action_output) == 2:
            return action_output
        return action_output, None

    @staticmethod
    def _config(discretization: dict | None):
        if discretization is None:
            return None
        return {**DEFAULT_DISCRETIZATION, **discretization}


class DiscretizedObservationEnv(gym.ObservationWrapper):
    """Wrap a Box-observation environment with tabular discrete observations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: gym.Env | str,
        discretization: dict | None = None,
        env_kwargs: dict | None = None,
        **make_kwargs,
    ):
        if isinstance(env, str):
            kwargs = dict(env_kwargs or {})
            kwargs.update(make_kwargs)
            env = gym.make(env, **kwargs)
        elif make_kwargs:
            raise TypeError(
                "Additional environment kwargs can only be used when env is an ID."
            )

        super().__init__(env)
        self.discretizer = RLEnvironmentRunner(
            self.env,
            discretization=discretization or DEFAULT_DISCRETIZATION,
        )
        if not self.discretizer.discrete_space:
            raise TypeError(
                "DiscretizedObservationEnv requires a Box observation space."
            )

        self.discrete_space = self.discretizer.discrete_space
        self.observation_space = gym.spaces.MultiDiscrete(
            np.array([len(space) for space in self.discrete_space], dtype=np.int64)
        )

    def observation(self, observation):
        """Return the observation as per-feature discrete bin indices."""
        return np.asarray(self.discretizer.get_discrete(observation), dtype=np.int64)


def register_discretized_env(
    env: gym.Env | str,
    id: str | None = None,
    discretization: dict | None = None,
    env_kwargs: dict | None = None,
    *,
    force: bool = False,
    **registration_kwargs,
) -> str:
    """
    Register a local Gymnasium environment with discretized observations.

    ``env`` can be an environment ID such as ``"CartPole-v1"`` or an existing
    Gymnasium environment. When an environment instance has a spec, the
    registration uses its spec ID so each ``gym.make`` call creates a fresh base
    environment.
    """
    env_source, resolved_env_kwargs = _registration_env_source(env, env_kwargs)
    env_id = id or _default_discretized_env_id(env)

    if env_id in registry and not force:
        return env_id
    if force and env_id in registry:
        del registry[env_id]

    register(
        id=env_id,
        entry_point=DiscretizedObservationEnv,
        kwargs={
            "env": env_source,
            "discretization": discretization,
            "env_kwargs": resolved_env_kwargs,
        },
        **registration_kwargs,
    )
    return env_id


def _registration_env_source(
    env: gym.Env | str,
    env_kwargs: dict | None = None,
) -> tuple[gym.Env | str, dict | None]:
    if isinstance(env, str):
        return env, dict(env_kwargs or {})

    spec = getattr(env, "spec", None)
    if spec is None or spec.id is None:
        return env, dict(env_kwargs or {}) if env_kwargs is not None else None

    kwargs = dict(getattr(spec, "kwargs", {}) or {})
    if spec.max_episode_steps is not None:
        kwargs.setdefault("max_episode_steps", spec.max_episode_steps)
    if env_kwargs is not None:
        kwargs.update(env_kwargs)
    return spec.id, kwargs


def _default_discretized_env_id(env: gym.Env | str) -> str:
    if isinstance(env, str):
        base_id = env
    else:
        spec = getattr(env, "spec", None)
        base_id = spec.id if spec is not None and spec.id is not None else None

    if base_id is None:
        base_name = env.unwrapped.__class__.__name__
    else:
        base_name = base_id.split("/")[-1]
        version_prefix, _, version = base_name.rpartition("-v")
        if version_prefix and version.isdigit():
            base_name = version_prefix
    return f"rl_suite/Discretized{base_name}-v0"


class HumanEnvironmentRunner:
    """Gymnasium adapter that prompts a human for an action at every step."""

    def __init__(self, env: gym.Env):
        self.env = env

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    def run_episode(
        self,
        *,
        seed: int | None = None,
        close_env: bool = False,
        input_fn: Callable[[str], str] = input,
        action_prompt: str | Callable[[gym.Env], str] | None = None,
        invalid_action_message: Callable[[str], str] | None = None,
        step_message: Callable | None = None,
        final_message: Callable | None = None,
        print_messages: bool = True,
        render_fn: Callable | None = None,
        render_each_step: bool = True,
        store_initial_state: bool = False,
    ) -> list[tuple]:
        """Prompt for actions until the episode terminates, truncates, or exits."""
        episode: list[tuple] = []
        reset_kwargs = {} if seed is None else {"seed": seed}
        state, _ = self.env.reset(**reset_kwargs)
        terminated = truncated = exited = False

        while not (terminated or truncated or exited):
            if render_each_step and render_fn is not None:
                render_fn(self)
            elif render_each_step:
                self.env.render()

            prompt = self._action_prompt(action_prompt)
            user_input = input_fn(prompt)
            if user_input.lower() == "exit":
                exited = True
                break

            try:
                action = self._parse_action(user_input)
            except ValueError:
                message = (
                    invalid_action_message(user_input)
                    if invalid_action_message is not None
                    else f"Invalid input: {user_input!r}."
                )
                if print_messages:
                    print(message)
                continue

            next_state, reward, terminated, truncated, info = self.env.step(action)
            stored_state = state if store_initial_state else next_state
            episode.append((stored_state, action, reward))
            if step_message is not None and print_messages:
                print(step_message(action, next_state, reward, info))
            state = next_state

        if final_message is not None and print_messages:
            print(final_message(state, terminated, truncated, exited))
        if close_env:
            self.env.close()
        return episode

    execute_environment = run_episode
    run = run_episode

    def _action_prompt(self, action_prompt):
        if callable(action_prompt):
            return action_prompt(self.env)
        if action_prompt is not None:
            return action_prompt
        return f"Action ({self.env.action_space}) or 'exit': "

    def _parse_action(self, user_input: str):
        space = self.env.action_space
        if isinstance(space, gym.spaces.Discrete):
            action = int(user_input)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            action = np.fromstring(user_input, sep=" ", dtype=int)
        elif isinstance(space, gym.spaces.Box):
            action = np.fromstring(user_input, sep=" ", dtype=float).reshape(
                space.shape
            )
        else:
            action = int(user_input)

        if hasattr(space, "contains") and not space.contains(action):
            raise ValueError(f"Action {action!r} is not in {space}")
        return action


def get_spaces_from_env(env):
    return RLEnvironmentRunner.from_env(env).get_spaces()


def neat_int(arr) -> list[int]:
    return [int(x) for x in arr]


def print_discrete_space(list_of_spaces) -> None:
    names = [
        "Cart Position",
        "Cart Velocity",
        "Pole Angle",
        "Pole Angular Velocity",
    ]
    for i, space in enumerate(list_of_spaces):
        print(f"{names[i]}: {space}\n")


__all__ = [
    "DEFAULT_DISCRETIZATION",
    "DiscretizedObservationEnv",
    "HumanEnvironmentRunner",
    "RLEnvironmentRunner",
    "discretize_interval",
    "even_bin_count",
    "get_spaces_from_env",
    "neat_int",
    "print_discrete_space",
    "register_discretized_env",
]
