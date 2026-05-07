"""Reusable reinforcement-learning algorithms and experiment helpers."""

from importlib import import_module


_PUBLIC_SUBMODULES = frozenset({
    "algorithms",
    "callbacks",
    "dp",
    "environments",
    "experiments",
    "mc",
    "td",
    "utils",
    "visualization",
})

__all__ = sorted(_PUBLIC_SUBMODULES)


def __getattr__(name: str):
    if name in _PUBLIC_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
