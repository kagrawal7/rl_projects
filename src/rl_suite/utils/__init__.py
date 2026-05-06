"""Utilities package exports.

This module auto-re-exports public symbols from selected submodules so the
package API stays in sync when functions are added/removed in those modules.
"""

from importlib import import_module

_EXPORT_MODULES = (
    "environment",
    "rendering",
)

__all__: list[str] = []
for _module_name in _EXPORT_MODULES:
    _module = import_module(f"{__name__}.{_module_name}")
    _public_names = getattr(
        _module, "__all__", [name for name in vars(_module) if not name.startswith("_")]
    )
    for _name in _public_names:
        globals()[_name] = getattr(_module, _name)
    __all__.extend(_public_names)

__all__ = sorted(set(__all__))
