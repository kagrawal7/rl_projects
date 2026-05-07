import gymnasium as gym

import rl_suite
import rl_suite.algorithms as algorithms
import rl_suite.callbacks as callbacks
from rl_suite import dp, mc, td
from rl_suite.algorithms.dp.policy_iteration import PolicyIteration
from rl_suite.algorithms.mc.off_policy import OffPolicyMonteCarlo
from rl_suite.algorithms.td.q_learning import QLearning
from rl_suite.environments import DiscretizedObservationEnv
from rl_suite.td import SARSA as SARSAFromFamily
from rl_suite.visualization import render_env_in_notebook


def test_direct_algorithm_imports_have_public_class_names():
    assert algorithms.QLearning.__name__ == "QLearning"
    assert algorithms.SARSA.__name__ == "SARSA"
    assert algorithms.ExpectedSARSA.__name__ == "ExpectedSARSA"
    assert algorithms.ValueIteration.__name__ == "ValueIteration"


def test_public_algorithm_classes_are_the_implementations():
    assert algorithms.QLearning is QLearning
    assert algorithms.PolicyIteration is PolicyIteration
    assert algorithms.OffPolicyMonteCarlo is OffPolicyMonteCarlo


def test_algorithm_family_modules_expose_discovery_helpers():
    assert td.get_implemented_algorithms() == [
        "QLearning",
        "SARSA",
        "ExpectedSARSA",
    ]
    assert td.get_implemented_algs() == td.get_implemented_algorithms()
    assert mc.get_implemented_algorithms() == ["OffPolicyMonteCarlo"]
    assert dp.get_implemented_algorithms() == [
        "PolicyIteration",
        "StochasticPolicyEvaluation",
        "ValueIteration",
    ]
    assert algorithms.get_implemented_algorithms() == [
        "PolicyIteration",
        "StochasticPolicyEvaluation",
        "ValueIteration",
        "OffPolicyMonteCarlo",
        "QLearning",
        "SARSA",
        "ExpectedSARSA",
    ]


def test_top_level_package_exposes_lowercase_family_modules():
    assert rl_suite.td.QLearning is algorithms.QLearning
    assert rl_suite.mc.OffPolicyMC is algorithms.OffPolicyMC
    assert rl_suite.dp.ValueIteration is algorithms.ValueIteration


def test_flat_algorithm_module_reexports_family_classes():
    assert rl_suite.algorithms.SARSA is td.SARSA
    assert algorithms.SARSA is td.SARSA
    assert algorithms.ValueIteration is dp.ValueIteration
    assert algorithms.OffPolicyMonteCarlo is mc.OffPolicyMonteCarlo


def test_family_modules_support_algorithm_imports():
    assert SARSAFromFamily is algorithms.SARSA


def test_public_helpers_are_importable_from_domain_modules():
    assert DiscretizedObservationEnv is not None
    assert callable(render_env_in_notebook)
    assert callbacks.CallbackList is not None


def test_public_algorithms_can_be_instantiated():
    env = gym.make("FrozenLake-v1")
    try:
        agent = algorithms.QLearning(env)
        assert agent.env is env
    finally:
        env.close()
