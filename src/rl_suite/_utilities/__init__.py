from importlib import import_module


_ENVIRONMENT_EXPORTS = frozenset({
    "DEFAULT_DISCRETIZATION",
    "DiscretizedObservationEnv",
    "HumanEnvironmentRunner",
    "ObservationDiscretizer",
    "discretize_interval",
    "even_bin_count",
    "get_discrete_state",
    "get_spaces_from_env",
    "get_state_shape",
    "neat_int",
    "print_discrete_space",
    "register_discretized_env",
    "run_episode",
})

_RENDERING_EXPORTS = frozenset({
    "display_renders",
    "render_env_in_notebook",
})

__all__ = sorted(
    _ENVIRONMENT_EXPORTS
    | _RENDERING_EXPORTS
    | {"_Utilities", "environment", "rendering"}
)


def __getattr__(name: str):
    if name == "environment":
        module = import_module(f"{__name__}.environment")
    elif name == "rendering":
        module = import_module(f"{__name__}.rendering")
    elif name == "_Utilities":
        module = import_module(f"{__name__}._utils_class")
        value = module._Utilities
        globals()[name] = value
        return value
    elif name in _ENVIRONMENT_EXPORTS:
        module = import_module(f"{__name__}.environment")
        value = getattr(module, name)
        globals()[name] = value
        return value
    elif name in _RENDERING_EXPORTS:
        module = import_module(f"{__name__}.rendering")
        value = getattr(module, name)
        globals()[name] = value
        return value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = module
    return module
