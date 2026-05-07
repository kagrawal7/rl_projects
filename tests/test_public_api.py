import importlib

import gymnasium as gym
import pytest

import rl_suite
import rl_suite.callbacks as callbacks
from rl_suite import dp, mc, td
from rl_suite.algos import QLearning, SARSA, ExpectedSARSA, ValueIteration
from rl_suite.environments import DiscretizedObservationEnv
from rl_suite.td import SARSA as SARSAFromFamily
from rl_suite.visualization import render_env_in_notebook


def test_direct_algorithm_imports_have_public_class_names():
    assert QLearning.__name__ == "QLearning"
    assert SARSA.__name__ == "SARSA"
    assert ExpectedSARSA.__name__ == "ExpectedSARSA"
    assert ValueIteration.__name__ == "ValueIteration"


def test_algorithm_family_modules_expose_discovery_helpers():
    assert td.get_implemented_algos() == [
        "QLearning",
        "SARSA",
        "ExpectedSARSA",
    ]
    assert td.get_implemented_algs() == td.get_implemented_algorithms()
    assert mc.get_implemented_algos() == ["OffPolicyMonteCarlo"]
    assert dp.get_implemented_algos() == [
        "PolicyIteration",
        "StochasticPolicyEvaluation",
        "ValueIteration",
    ]


def test_top_level_package_exposes_lowercase_family_modules():
    assert rl_suite.td.QLearning is QLearning
    assert rl_suite.mc.OffPolicyMC.__name__ == "OffPolicyMonteCarlo"
    assert rl_suite.dp.ValueIteration is ValueIteration


def test_family_modules_support_algorithm_imports():
    assert SARSAFromFamily is SARSA


def test_public_helpers_are_importable_from_domain_modules():
    assert DiscretizedObservationEnv is not None
    assert callable(render_env_in_notebook)
    assert callbacks.CallbackList is not None


def test_public_algorithms_can_be_instantiated():
    env = gym.make("FrozenLake-v1")
    try:
        agent = QLearning(env)
        assert agent.env is env
    finally:
        env.close()


def test_removed_toolbox_namespace_is_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rl_suite.RLToolbox")
