from .base import Callback


class HistoryCallback(Callback):
    """Store every callback event as a flat record."""

    def __init__(self):
        self.history = []

    def reset(self):
        self.history = []
        return self

    def on_train_begin(self, data=None):
        self.record("train_begin", data)

    def on_train_end(self, data=None):
        self.record("train_end", data)

    def on_episode_begin(self, data=None):
        self.record("episode_begin", data)

    def on_episode_end(self, data=None):
        self.record("episode_end", data)

    def on_step(self, data=None):
        self.record("step", data)

    def on_update(self, data=None):
        self.record("update", data)

    def record(self, event, data=None):
        row = {"event": event}
        row.update(data or {})
        self.history.append(row)


class EpisodeReturnCallback(Callback):
    """Track episode returns from step rewards or episode-end data."""

    def __init__(self):
        self.returns = []
        self.lengths = []
        self.current_return = 0
        self.current_length = 0

    def reset(self):
        self.returns = []
        self.lengths = []
        self.current_return = 0
        self.current_length = 0
        return self

    def on_episode_begin(self, data=None):
        self.current_return = 0
        self.current_length = 0

    def on_step(self, data=None):
        data = data or {}
        self.current_return += data.get("reward", 0)
        self.current_length += 1

    def on_episode_end(self, data=None):
        data = data or {}
        episode_return = data.get("return", data.get("episode_return", self.current_return))
        episode_length = data.get("length", data.get("episode_length", self.current_length))
        self.returns.append(episode_return)
        self.lengths.append(episode_length)


class QDeltaCallback(Callback):
    """Collect q_delta values from update events."""

    def __init__(self):
        self.q_deltas = []

    def reset(self):
        self.q_deltas = []
        return self

    def on_update(self, data=None):
        data = data or {}
        if "q_delta" in data:
            self.q_deltas.append(data["q_delta"])


class EpsilonCallback(Callback):
    """Collect epsilon values from update or episode-end events."""

    def __init__(self):
        self.epsilons = []

    def reset(self):
        self.epsilons = []
        return self

    def on_update(self, data=None):
        self._record(data)

    def on_episode_end(self, data=None):
        self._record(data)

    def _record(self, data=None):
        data = data or {}
        if "epsilon" in data:
            self.epsilons.append(data["epsilon"])
