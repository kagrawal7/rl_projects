from ._gpi._gpi import _GPI
from ._mc._mc import _MC
from ._td._td import _TD


class _Algorithms:

    def __init__(self):
        self._init_agents()

    def _init_agents(self):
        self.gpi = _GPI()
        self.mc = _MC()
        self.td = _TD()
