import gymnasium as gym

import rl_suite as rl
import rl_suite.algorithms as algs


def test_algorithms_module_exposes_environment_agnostic_namespaces():
    env1 = gym.make("FrozenLake-v1")
    env2 = gym.make("FrozenLake-v1")

    try:
        caller1 = algs.td.sarsa(env1)
        caller2 = algs.td.sarsa(env2)

        assert rl.algorithms is algs
        assert caller1.env is env1
        assert caller2.env is env2
        assert caller1 is not caller2
    finally:
        env1.close()
        env2.close()


def test_rl_toolbox_exposes_same_algorithm_namespace_without_binding_env():
    env = gym.make("FrozenLake-v1")

    try:
        toolbox = rl.RLToolbox(env)
        agent = toolbox.algorithms.td.q_learning(env)

        assert rl.RLToolbox.algorithms is algs
        assert toolbox.algorithms is algs
        assert agent.env is env
    finally:
        env.close()
