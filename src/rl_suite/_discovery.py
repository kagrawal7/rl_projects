"""Small helpers for discovering public algorithm classes."""

from __future__ import annotations

import inspect
import sys


def discover_algorithm_classes(module_name: str, base_class: type) -> list[type]:
    """Return public classes defined in ``module_name`` that subclass ``base_class``."""
    module = sys.modules[module_name]
    classes = []
    for name, value in module.__dict__.items():
        if not inspect.isclass(value):
            continue
        if not (
            value.__module__ == module_name
            or value.__module__.startswith(f"{module_name}.")
        ):
            continue
        if name.startswith("_") or name != value.__name__:
            continue
        if not issubclass(value, base_class) or value is base_class:
            continue
        classes.append(value)
    return classes


def class_names(classes: list[type]) -> list[str]:
    """Return class names in discovery order."""
    return [cls.__name__ for cls in classes]


__all__ = ["class_names", "discover_algorithm_classes"]
