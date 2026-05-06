from abc import ABC, abstractmethod
import gymnasium as gym

class AbstractAgent(ABC):
    """ An abstract class to be used as a bass class for agents """
    
    def get_env_info(self, env):
        n, k = env.observation_space.n, env.action_space.n
        S, A = list(range(n)), list(range(k))
        self.S_plus, self.A, self.n, self.k = S, A, n, k

    @abstractmethod
    def select_action(self, state):
        pass
