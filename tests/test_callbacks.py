import rl_suite.RLToolbox.callbacks as callbacks


def test_history_callback_records_events():
    logger = callbacks.HistoryCallback()

    logger.on_step({"state": 1, "action": 0, "reward": 1})
    logger.on_update({"q_delta": 0.5})

    assert logger.history == [
        {"event": "step", "state": 1, "action": 0, "reward": 1},
        {"event": "update", "q_delta": 0.5},
    ]


def test_episode_return_callback_can_accumulate_step_rewards():
    logger = callbacks.EpisodeReturnCallback()

    logger.on_episode_begin()
    logger.on_step({"reward": 1})
    logger.on_step({"reward": -0.25})
    logger.on_episode_end()

    assert logger.returns == [0.75]
    assert logger.lengths == [2]


def test_callback_list_forwards_hooks():
    q_logger = callbacks.QDeltaCallback()
    epsilon_logger = callbacks.EpsilonCallback()
    logger = callbacks.CallbackList([q_logger, epsilon_logger])

    logger.on_update({"q_delta": 0.2, "epsilon": 0.1})

    assert q_logger.q_deltas == [0.2]
    assert epsilon_logger.epsilons == [0.1]


def test_experiment_callbacks_store_training_and_rollout_metrics():
    callbacks_ = callbacks.CallbackList([
        callbacks.TimerCallback(),
        callbacks.ConvergenceCallback(),
        callbacks.ValueFunctionCallback(),
        callbacks.PolicyCallback(),
        callbacks.RolloutCallback(),
    ])

    callbacks_.on_train_begin()
    callbacks_.on_update({
        "delta": 0.1,
        "sweep": 1,
        "values": [0, 1],
        "policy": [1, 0],
    })
    callbacks_.on_train_end({
        "values": [0.5, 1.0],
        "policy": [0, 1],
    })
    callbacks_.on_episode_end({
        "return": 1,
        "length": 3,
        "success": True,
        "path": [0, 1, 2],
    })

    timer, convergence, values, policy, rollouts = callbacks_.callbacks
    assert timer.elapsed is not None
    assert convergence.deltas == [0.1]
    assert values.final_values == [0.5, 1.0]
    assert policy.final_policy == [0, 1]
    assert rollouts.success_rate == 1
