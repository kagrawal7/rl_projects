__all__ = ["AbstractAgent", "_GPIBaseAgent", "_TDBaseAgent"]


def __getattr__(name):
    if name == "AbstractAgent":
        from ._agent_base import AbstractAgent

        return AbstractAgent
    if name == "_GPIBaseAgent":
        from ._gpi_base import _GPIBaseAgent

        return _GPIBaseAgent
    if name == "_TDBaseAgent":
        from ._td_base import _TDBaseAgent

        return _TDBaseAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
