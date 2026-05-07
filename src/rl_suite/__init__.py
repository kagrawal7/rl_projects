"""Reusable reinforcement-learning algorithms and experiment helpers."""

from importlib import import_module
import sys

from rl_suite import algorithms
from rl_suite.algorithms import dp, mc, td


_ALIASED_SUBMODULES = {
    "algorithms": algorithms,
    "dp": dp,
    "mc": mc,
    "td": td,
}

for _name, _module in _ALIASED_SUBMODULES.items():
    sys.modules[f"{__name__}.{_name}"] = _module


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
    if name in _ALIASED_SUBMODULES:
        return _ALIASED_SUBMODULES[name]
    if name in _PUBLIC_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
