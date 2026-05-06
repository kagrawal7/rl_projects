from __future__ import annotations

from collections.abc import Callable

from .signal_expeditor import SignalExpeditor, discretize_interval, even_bin_count


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


def run_cartpole_episode(
    env,
    select_action: Callable,
    behaviour=None,
    *,
    close_env: bool = True,
):
    """
    Roll out one CartPole episode for tabular control.

    - If ``behaviour`` is None, each step uses ``select_action(state)`` and the
      transition is stored as ``(state, action, reward)``.
    - Otherwise uses ``select_action(state, behaviour)`` and stores
      ``(state, action, prob, reward)`` (importance sampling for off-policy MC).
    """
    expeditor = SignalExpeditor.from_env(env)
    return expeditor.execute_environment(
        select_action,
        behaviour,
        close_env=close_env,
    )


__all__ = [
    "SignalExpeditor",
    "discretize_interval",
    "even_bin_count",
    "neat_int",
    "print_discrete_space",
    "run_cartpole_episode",
]
