from .gpi.gpi_agent import _GPIAgent
from .mc.mc_agent import _MCAgent

class Algorithms:
    """Main client for interacting with the AI Library API."""

    def __init__(self):
        # Initialize resources
        self.gpi = _GPIAgent()
        self.mc = _MCAgent()
        