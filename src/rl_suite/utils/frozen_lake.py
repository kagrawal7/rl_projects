from __future__ import annotations

from collections.abc import Callable

from .rendering import render_env_in_notebook
from .signal_expeditor import SignalExpeditor


def run_frozen_lake_episode(
    env,
    agent=None,
    seed: int = 10,
    suppress_text: bool = True,
    goal_state: int = 15,
    close_env: bool = True,
    action_message: str | None = None,
    invalid_action_message: Callable[[str], str] | None = None,
    step_message: Callable | None = None,
    final_message: Callable | None = None,
    render_each_step: bool | None = None,
):
    """
    Run one episode with FrozenLake-friendly default messages.

    Environment execution is handled by ``SignalExpeditor``; this wrapper only
    supplies optional text/render defaults for older FrozenLake notebooks.
    """
    action_message = action_message or (
        "Possible actions: [0: left, 1: down, 2: right, 3: up], "
        "or 'exit' to stop program"
    )
    invalid_action_message = invalid_action_message or (
        lambda user_input: (
            f"Invalid input: {user_input!r}. "
            "Please enter an integer action or 'exit'."
        )
    )

    def default_step_message(action, state, reward, info) -> str:
        prob = info.get("prob", "N/A")
        return (
            f"--> The result of taking action {action} is:\n"
            f"     S = {state}\n"
            f"     R = {reward}\n"
            f"     p = {prob}"
        )

    def default_final_message(state, terminated, truncated, exited) -> str:
        if exited:
            termination = "Episode has been terminated by user."
        elif terminated:
            termination = "Episode has terminated!"
        elif truncated:
            termination = "Episode has finished due to truncate flag."
        else:
            termination = "Episode stopped."

        result = (
            "Yay! You reached the goal!"
            if state == goal_state
            else "You did not reach the goal."
        )
        return f"{termination}\nFinal state of environment:\n{result}"

    step_message = step_message or default_step_message
    final_message = final_message or default_final_message
    render_each_step = agent is None if render_each_step is None else render_each_step

    return SignalExpeditor.from_env(env).execute_environment(
        agent=agent,
        seed=seed,
        close_env=close_env,
        human=agent is None,
        action_message=action_message,
        invalid_action_message=invalid_action_message,
        step_message=step_message,
        final_message=final_message,
        show_messages=not suppress_text,
        render_fn=render_env_in_notebook,
        render_each_step=render_each_step,
    )


__all__ = ["SignalExpeditor", "run_frozen_lake_episode"]
