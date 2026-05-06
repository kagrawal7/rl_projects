"""Deprecated compatibility exports for environment execution helpers."""

from .frozen_lake import run_frozen_lake_episode
from .signal_expeditor import SignalExpeditor

execute_environment = run_frozen_lake_episode


def get_spaces_from_env(env):
    return SignalExpeditor.from_env(env).get_spaces()


__all__ = [
    "SignalExpeditor",
    "execute_environment",
    "get_spaces_from_env",
    "run_frozen_lake_episode",
]
