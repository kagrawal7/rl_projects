from __future__ import annotations

from .rendering import display_renders, render_env_in_notebook


def run_frozen_lake_episode(
    env,
    agent=None,
    seed: int = 10,
    suppress_text: bool = True,
    goal_state: int = 15,
    close_env: bool = True,
):
    """
    Run one FrozenLake (or compatible discrete) episode: human input or an agent.
    """

    def log(message: str) -> None:
        if not suppress_text:
            print(message)

    def step_message(action, state, reward, info) -> str:
        prob = info.get("prob", "N/A")
        return (
            f"--> The result of taking action {action} is:\n"
            f"     S = {state}\n"
            f"     R = {reward}\n"
            f"     p = {prob}"
        )

    def result_message(reached_goal: bool) -> str:
        return (
            "Yay! You reached the goal!"
            if reached_goal
            else "You did not reach the goal."
        )

    actions_message = (
        "Possible actions: [0: left, 1: down, 2: right, 3: up], "
        "or 'exit' to stop program"
    )

    images: list = []
    termination_message = "Episode has "

    state, info = env.reset(seed=seed)

    human_agent = agent is None
    if agent is not None:
        agent.get_env_info(env)

    terminated = False
    truncated = False
    exited_by_user = False

    while not (terminated or truncated):
        if human_agent:
            render_env_in_notebook(env)
            log(actions_message)

            user_input = input()

            if user_input.lower() == "exit":
                termination_message += "been terminated by user."
                exited_by_user = True
                break

            try:
                action = int(user_input)
            except ValueError:
                log(
                    f"Invalid input: {user_input!r}. "
                    "Please enter an integer action or 'exit'."
                )
                continue

        else:
            images.append(env.render())
            action = agent.select_action(state)

        state, reward, terminated, truncated, info = env.step(action)
        log(step_message(action, state, reward, info))

    display_renders(images)

    if not exited_by_user:
        if terminated:
            termination_message += "terminated!"
        elif truncated:
            termination_message += "finished due to truncate flag."

    print(f"{termination_message}\nFinal state of environment:")
    render_env_in_notebook(env)
    print(result_message(state == goal_state))

    if close_env:
        env.close()
