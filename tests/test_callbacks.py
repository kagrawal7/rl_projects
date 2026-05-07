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
