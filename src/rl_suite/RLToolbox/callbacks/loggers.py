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


class TimerCallback(Callback):
    """Measure training wall time from train-begin/train-end hooks."""

    def __init__(self):
        self.elapsed = None
        self.started_at = None

    def reset(self):
        self.elapsed = None
        self.started_at = None
        return self

    def on_train_begin(self, data=None):
        from time import perf_counter

        self.started_at = perf_counter()

    def on_train_end(self, data=None):
        from time import perf_counter

        if self.started_at is not None:
            self.elapsed = perf_counter() - self.started_at


class ConvergenceCallback(Callback):
    """Track deltas and iteration/sweep counts from update events."""

    def __init__(self):
        self.deltas = []
        self.events = []
        self.iterations = 0
        self.sweeps = 0

    def reset(self):
        self.deltas = []
        self.events = []
        self.iterations = 0
        self.sweeps = 0
        return self

    def on_update(self, data=None):
        data = data or {}
        self.events.append(dict(data))
        if "delta" in data:
            self.deltas.append(data["delta"])
        if "iteration" in data:
            self.iterations = max(self.iterations, data["iteration"])
        if "sweep" in data:
            self.sweeps = max(self.sweeps, data["sweep"])


class ValueFunctionCallback(Callback):
    """Store value-function snapshots from update or train-end events."""

    def __init__(self, every=1):
        self.every = every
        self.snapshots = []
        self.final_values = None

    def reset(self):
        self.snapshots = []
        self.final_values = None
        return self

    def on_update(self, data=None):
        data = data or {}
        values = data.get("values")
        step = data.get("sweep", data.get("iteration", len(self.snapshots) + 1))
        if values is not None and step % self.every == 0:
            self.snapshots.append(values.copy() if hasattr(values, "copy") else values)

    def on_train_end(self, data=None):
        data = data or {}
        values = data.get("values")
        if values is not None:
            self.final_values = values.copy() if hasattr(values, "copy") else values


class PolicyCallback(Callback):
    """Store policy snapshots and the final policy."""

    def __init__(self):
        self.snapshots = []
        self.final_policy = None

    def reset(self):
        self.snapshots = []
        self.final_policy = None
        return self

    def on_update(self, data=None):
        data = data or {}
        policy = data.get("policy")
        if policy is not None:
            self.snapshots.append(policy.copy() if hasattr(policy, "copy") else policy)

    def on_train_end(self, data=None):
        data = data or {}
        policy = data.get("policy")
        if policy is not None:
            self.final_policy = policy.copy() if hasattr(policy, "copy") else policy


class RolloutCallback(Callback):
    """Track evaluation returns, lengths, success rate, and latest trajectory."""

    def __init__(self):
        self.returns = []
        self.lengths = []
        self.successes = []
        self.latest_path = []

    def reset(self):
        self.returns = []
        self.lengths = []
        self.successes = []
        self.latest_path = []
        return self

    def on_episode_end(self, data=None):
        data = data or {}
        if "return" in data:
            self.returns.append(data["return"])
        if "length" in data:
            self.lengths.append(data["length"])
        if "success" in data:
            self.successes.append(data["success"])
        if "path" in data:
            self.latest_path = data["path"]

    @property
    def success_rate(self):
        return sum(self.successes) / len(self.successes) if self.successes else 0
