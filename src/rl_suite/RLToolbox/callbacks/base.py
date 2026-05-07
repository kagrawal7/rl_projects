class Callback:
    """Base callback with no-op hooks."""

    def reset(self):
        return self

    def on_train_begin(self, data=None):
        pass

    def on_train_end(self, data=None):
        pass

    def on_episode_begin(self, data=None):
        pass

    def on_episode_end(self, data=None):
        pass

    def on_step(self, data=None):
        pass

    def on_update(self, data=None):
        pass


class CallbackList(Callback):
    def __init__(self, callbacks=None):
        self.callbacks = list(callbacks or [])

    def append(self, callback):
        self.callbacks.append(callback)

    def reset(self):
        for callback in self.callbacks:
            callback.reset()
        return self

    def on_train_begin(self, data=None):
        self._call("on_train_begin", data)

    def on_train_end(self, data=None):
        self._call("on_train_end", data)

    def on_episode_begin(self, data=None):
        self._call("on_episode_begin", data)

    def on_episode_end(self, data=None):
        self._call("on_episode_end", data)

    def on_step(self, data=None):
        self._call("on_step", data)

    def on_update(self, data=None):
        self._call("on_update", data)

    def _call(self, hook, data):
        for callback in self.callbacks:
            getattr(callback, hook)(data)
