from abc import abstractmethod
from ...agent_base import AbstractAgent


class _GPIBase(AbstractAgent):
    """Base class for an agent that uses a policy function to select an action."""

    def __init__(self, gamma, theta):
        self.gamma = gamma
        self.theta = theta
        self.no_policy_set = True

    def _expected_return(self, action_vector):
        """Helper (summation over s' and r given transition tuple)."""
        s_prime, reward = action_vector[1:3]
        return reward + self.gamma * self.V[s_prime]

    def _action_argmax(self, s):
        """Return action that maximizes expected return for given ``s``."""
        state_obj = self.p[s]
        action = max(state_obj, key=lambda a: self._expected_return(state_obj[a][0]))
        return action

    def get_env_info(self, env):
        super().get_env_info(env)
        self.p = env.unwrapped.P
        self.V = []
        self.terminals = set()
        for s in range(self.n):
            for action_key in self.p[s]:
                vec = self.p[s][action_key][0]
                if vec[-1]:
                    self.V.append(0)
                    self.terminals.add(vec[1])
                else:
                    self.V.append(1)
        if self.no_policy_set:
            self._intialize_policy()

    def policy_evaluation(self, print_num_iter=False):
        """Evaluate policy using dynamic programming policy evaluation."""
        if self.theta <= 0:
            raise ValueError("Theta must be positive number!")
        num_iter = 0
        while True:
            delta = 0
            num_iter += 1
            for s in range(self.n):
                if s in self.terminals:
                    continue
                old_val = self.V[s]
                self.V[s] = self._value_update(s)
                delta = max(delta, abs(old_val - self.V[s]))
            if delta < self.theta:
                break
        if print_num_iter:
            print(
                f"theta={self.theta} and gamma={self.gamma} ====> "
                f"number of steps in evalution: {num_iter}"
            )
        return self.policy, self.V

    @abstractmethod
    def _intialize_policy(self):
        pass

    @abstractmethod
    def _value_update(self, s):
        pass

    @abstractmethod
    def select_action(self, state):
        pass
