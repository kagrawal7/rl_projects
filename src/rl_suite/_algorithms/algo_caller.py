from ._gpi._gpi import _GPI
from ._mc._mc import _MC
from ._td._td import _TD


class Algorithms:
    """Main client for interacting with the AI Library API."""

    def __init__(self, env):
        # Initialize resources
        self.set_env(env)

    def _init_agents(self):
        env = self.env
        self.gpi = _GPI(env)
        self.mc = _MC(env)
        self.td = _TD(env)
    
    def set_env(self, new_env):
        self.env = new_env
        self._init_agents()