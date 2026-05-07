import gymnasium as gym

from rl_suite.utils import DiscretizedObservationEnv, register_discretized_env


CARTPOLE_DISCRETIZATION = {
    "num_bins": 10,
    "intervals": [(-4.8, 4.8), (-2.5, 2.5), (-0.418, 0.418), (-3.5, 3.5)],
}


def test_discretized_observation_env_wraps_cartpole_observations():
    env = DiscretizedObservationEnv(
        "CartPole-v1",
        discretization=CARTPOLE_DISCRETIZATION,
    )

    observation, _ = env.reset(seed=10)

    assert env.observation_space.shape == (4,)
    assert env.observation_space.contains(observation)

    next_observation, _, _, _, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(next_observation)
    env.close()


def test_register_discretized_env_makes_local_gymnasium_env():
    env_id = register_discretized_env(
        "CartPole-v1",
        id="rl_suite/TestDiscretizedCartPole-v0",
        discretization=CARTPOLE_DISCRETIZATION,
        force=True,
    )

    env = gym.make(env_id)
    observation, _ = env.reset(seed=10)

    assert env_id == "rl_suite/TestDiscretizedCartPole-v0"
    assert env.observation_space.shape == (4,)
    assert env.observation_space.contains(observation)

    env.close()
