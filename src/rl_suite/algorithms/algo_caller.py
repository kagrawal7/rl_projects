from .gpi.gpi_agent import _GPIAgent
from .mc.mc_agent import _MCAgent
from .td.td_agent import _TDAgent


class Algorithms:
    """Main client for interacting with the AI Library API."""

    def __init__(self):
        # Initialize resources
        self.gpi = _GPIAgent()
        self.mc = _MCAgent()
        self.td = _TDAgent()
        