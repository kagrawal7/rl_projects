from __future__ import annotations

from collections.abc import Callable

import numpy as np


def even_bin_count(n: int, min_val: int = 6) -> int:
    """Return an even integer bin count at least ``min_val`` (matches notebook helpers)."""
    n = int(n)
    if n < min_val:
        raise ValueError(f"Cannot specify number smaller than {min_val}")
    return n + 1 if n % 2 else n


def discretize_interval(n: int, interval: tuple[float, float]) -> np.ndarray:
    """Symmetric bins around 0 for one scalar observation interval."""
    l, u = interval
    mid = n // 2
    lower = np.linspace(l, 0, mid, endpoint=False)
    upper = np.linspace(0, u, mid + 1)
    return np.concatenate([lower, upper])


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
    episode: list = []
    state, info = env.reset()
    finished = False
    while not finished:
        if behaviour is None:
            action = select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
        else:
            out = select_action(state, behaviour)
            if isinstance(out, tuple) and len(out) == 2:
                action, prob = out
                state, reward, terminated, truncated, info = env.step(action)
                episode.append((state, action, prob, reward))
            else:
                action = out
                state, reward, terminated, truncated, info = env.step(action)
                episode.append((state, action, reward))
        finished = terminated or truncated
    if close_env:
        env.close()
    return episode
