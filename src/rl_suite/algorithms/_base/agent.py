from abc import ABC, abstractmethod


class _Agent(ABC):
    """An abstract class to be used as a base class for agents."""

    def __init__(self, env):
        super().__init__()
        self.env = env

    def get_env_info(self, env):
        if hasattr(env, "get_spaces"):
            S, A, n, k = env.get_spaces()
        else:
            n, k = env.observation_space.n, env.action_space.n
            S, A = list(range(n)), list(range(k))
        self.S_plus, self.A, self.n, self.k = S, A, n, k

    @abstractmethod
    def select_action(self, state):
        pass
