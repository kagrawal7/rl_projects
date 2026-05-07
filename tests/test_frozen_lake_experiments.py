import numpy as np

from rl_suite.experiments.frozen_lake import (
    FrozenLakeDP,
    make_env,
    run_experiments,
)


def test_action_value_sums_all_slippery_transitions():
    env = make_env("sparse")
    solver = FrozenLakeDP(env)
    values = np.arange(env.observation_space.n, dtype=float)

    expected = sum(
        prob * (reward + (0 if done else solver.gamma * values[next_state]))
        for prob, next_state, reward, done in env.unwrapped.P[0][0]
    )

    assert len(env.unwrapped.P[0][0]) == 3
    assert solver.q(0, 0, values) == expected


def test_frozen_lake_experiment_grid_runs():
    experiments = run_experiments(rewards=("sparse",), gammas=(0.9,), episodes=2)

    assert [exp["algorithm"] for exp in experiments] == [
        "Policy Iteration",
        "Value Iteration",
    ]
    assert all(exp["policy"].shape == (16,) for exp in experiments)
