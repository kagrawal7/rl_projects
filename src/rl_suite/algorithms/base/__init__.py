__all__ = ["AbstractAgent", "_GPIBaseAgent", "_TDBaseAgent"]


def __getattr__(name):
    if name == "AbstractAgent":
        from .agent import AbstractAgent

        return AbstractAgent
    if name == "_GPIBaseAgent":
        from .dynamic_programming import _GPIBaseAgent

        return _GPIBaseAgent
    if name == "_TDBaseAgent":
        from .temporal_difference import _TDBaseAgent

        return _TDBaseAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
