import importlib

import gymnasium as gym
import pytest

import rl_suite.RLToolbox as rl


def test_toolbox_module_exposes_environment_agnostic_algorithm_namespaces():
    env1 = gym.make("FrozenLake-v1")
    env2 = gym.make("FrozenLake-v1")

    try:
        caller1 = rl.algorithms.td.sarsa(env1)
        caller2 = rl.algorithms.td.sarsa(env2)

        assert caller1.env is env1
        assert caller2.env is env2
        assert caller1 is not caller2
    finally:
        env1.close()
        env2.close()


def test_algorithms_and_utils_are_not_public_package_modules():
    import rl_suite

    assert not hasattr(rl_suite, "algorithms")
    assert not hasattr(rl_suite, "utils")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rl_suite.algorithms")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rl_suite.utils")
