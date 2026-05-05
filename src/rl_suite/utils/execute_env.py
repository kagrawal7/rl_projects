"""Deprecated: use ``rl_suite.utils.frozen_lake.run_frozen_lake_episode``."""

from .frozen_lake import run_frozen_lake_episode

execute_environment = run_frozen_lake_episode

__all__ = ["execute_environment", "run_frozen_lake_episode"]
