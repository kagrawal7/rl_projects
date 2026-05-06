from ._gpi._gpi_agent import _GPIAgent
from ._mc._mc_agent import _MCAgent
from ._td._td_agent import _TDAgent


class Algorithms:
    """Main client for interacting with the AI Library API."""

    def __init__(self):
        # Initialize resources
        self.gpi = _GPIAgent()
        self.mc = _MCAgent()
        self.td = _TDAgent()
        