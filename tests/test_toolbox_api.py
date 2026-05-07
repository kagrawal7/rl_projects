import importlib

import gymnasium as gym
import pytest

import rl_suite.RLToolbox.algorithms as rl
import rl_suite.RLToolbox.utilities as utils


def test_toolbox_module_exposes_environment_agnostic_algorithm_namespaces():
    env1 = gym.make("FrozenLake-v1")
    env2 = gym.make("FrozenLake-v1")

    try:
        caller1 = rl.td.sarsa(env1)
        caller2 = rl.td.sarsa(env2)

        assert caller1.env is env1
        assert caller2.env is env2
        assert caller1 is not caller2
    finally:
        env1.close()
        env2.close()


def test_toolbox_utilities_module_exposes_helpers():
    assert utils.DiscretizedObservationEnv is not None
    assert callable(utils.register_discretized_env)


def test_legacy_algorithm_and_utility_aliases_are_not_public_package_modules():
    import rl_suite

    assert not hasattr(rl_suite, "algorithms")
    assert not hasattr(rl_suite, "utilities")
    assert hasattr(rl_suite, "utils")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rl_suite.algorithms")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rl_suite.utilities")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rl_suite.RLToolbox.utils")
