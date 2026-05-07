from __future__ import annotations

from . import environment, rendering
from .environment import (
    DEFAULT_DISCRETIZATION,
    DiscretizedObservationEnv,
    HumanEnvironmentRunner,
    ObservationDiscretizer,
    discretize_interval,
    even_bin_count,
    get_discrete_state,
    get_spaces_from_env,
    get_state_shape,
    neat_int,
    print_discrete_space,
    register_discretized_env,
    run_episode,
)
from .rendering import display_renders, render_env_in_notebook


class _Utilities:
    """Facade for environment runners, discretization helpers, and rendering."""

    environment = environment
    rendering = rendering

    DEFAULT_DISCRETIZATION = DEFAULT_DISCRETIZATION
    DiscretizedObservationEnv = DiscretizedObservationEnv
    HumanEnvironmentRunner = HumanEnvironmentRunner
    ObservationDiscretizer = ObservationDiscretizer

    discretize_interval = staticmethod(discretize_interval)
    even_bin_count = staticmethod(even_bin_count)
    get_discrete_state = staticmethod(get_discrete_state)
    get_spaces_from_env = staticmethod(get_spaces_from_env)
    get_state_shape = staticmethod(get_state_shape)
    neat_int = staticmethod(neat_int)
    print_discrete_space = staticmethod(print_discrete_space)
    register_discretized_env = staticmethod(register_discretized_env)
    run_episode = staticmethod(run_episode)

    display_renders = staticmethod(display_renders)
    render_env_in_notebook = staticmethod(render_env_in_notebook)

    def __init__(self, env=None, discretization: dict | None = None):
        self.env = env
        self.discretization = discretization
        self.human_agent = None
        if env is not None:
            self.bind(env, discretization=discretization)

    def bind(self, env, discretization: dict | None = None) -> "_Utilities":
        self.env = env
        self.discretization = discretization
        self.human_agent = self.HumanEnvironmentRunner(env)
        return self

    def human(self, env=None) -> HumanEnvironmentRunner:
        env = self._resolve_env(env)
        return self.HumanEnvironmentRunner(env)

    def _resolve_env(self, env):
        if env is not None:
            return env
        if self.env is None:
            raise ValueError("No environment is bound to this utilities instance.")
        return self.env


__all__ = ["_Utilities"]
