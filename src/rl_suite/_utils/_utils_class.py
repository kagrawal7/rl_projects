
    # "HumanEnvironmentRunner",
    # "RLEnvironmentRunner",
    # "discretize_interval",
    # "even_bin_count",
    # "get_spaces_from_env",
    # "neat_int",
    # "print_discrete_space",

from .environment import HumanEnvironmentRunner
from .environment import RLEnvironmentRunner
from .environment import discretize_interval

class _Utilities:
    def __init__(self, env):
        self.human_agent = HumanEnvironmentRunner(env)
        self.rl_agent = RLEnvironmentRunner(env)

    # def discretize_interval(self, n: int, interval: tuple[float, float]) -> np.ndarray:
    #     """Create symmetric bins around 0 for one scalar observation interval."""
    #     lower_bound, upper_bound = interval
    #     mid = n // 2
    #     lower = np.linspace(lower_bound, 0, mid, endpoint=False)
    #     upper = np.linspace(0, upper_bound, mid + 1)
    #     return np.concatenate([lower, upper])