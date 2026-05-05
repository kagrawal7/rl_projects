"""Deprecated: use ``rl_suite.utils.frozen_lake.run_frozen_lake_episode``."""

from .frozen_lake import run_frozen_lake_episode

execute_environment = run_frozen_lake_episode

def get_spaces_from_env(env):
    n, k = env.observation_space.n, env.action_space.n
    S, A = list(range(n)), list(range(k))
    return S, A, n, k

__all__ = ["execute_environment", "run_frozen_lake_episode", "get_spaces_from_env"]
