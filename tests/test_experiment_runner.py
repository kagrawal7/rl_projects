import rl_suite.callbacks as callbacks
from rl_suite.experiments import GymExperiment, choose_action


class AlwaysLeft:
    def __init__(self, env, tag=None):
        self.env = env
        self.tag = tag
        self.trained = False

    def train(self, env):
        self.trained = env is self.env
        return ["trained"]

    def select_action(self, state):
        return 0


class PredictAgent:
    def predict(self, state):
        return 1, None


def test_gym_experiment_runs_registered_env_and_algorithm_configs():
    experiment = GymExperiment("FrozenLake-v1", AlwaysLeft, env_kwargs={
        "map_name": "4x4",
        "is_slippery": False,
    })
    config = {"name": "run 1", "agent": {"tag": "baseline"}}
    rows = experiment.run(configs=[config], eval_episodes=2, seed=4)

    assert len(rows) == 1
    assert rows[0]["agent"].tag == "baseline"
    assert rows[0]["agent"].trained
    assert rows[0]["history"] == ["trained"]
    assert rows[0]["config"] == config
    assert "mean_reward" in rows[0]


def test_gym_experiment_accepts_callback_factory():
    experiment = GymExperiment("FrozenLake-v1", AlwaysLeft, env_kwargs={
        "map_name": "4x4",
        "is_slippery": False,
    })

    rows = experiment.run(
        configs=[{"agent": {"tag": "one"}}, {"agent": {"tag": "two"}}],
        callbacks=lambda config: [callbacks.RolloutCallback()],
        eval_episodes=1,
        seed=4,
    )

    assert rows[0]["callbacks"] is not rows[1]["callbacks"]
    assert rows[0]["callbacks"].callbacks[0].lengths
    assert rows[1]["callbacks"].callbacks[0].lengths


def test_choose_action_supports_predict_style_agents():
    assert choose_action(PredictAgent(), 0) == 1
