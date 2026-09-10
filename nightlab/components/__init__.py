"""Component registry.

An experiment swaps one thing and keeps everything else fixed. That one thing
is usually a component: an attention layer, an MLP, a norm, or an optimizer.
Register a new component with the decorator and reference it by name from the
experiment YAML. Never edit the baseline components in place.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

_REGISTRY: dict[str, dict[str, Callable]] = defaultdict(dict)


def register(kind: str, name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        if name in _REGISTRY[kind]:
            raise ValueError(f"{kind}/{name} already registered")
        _REGISTRY[kind][name] = fn
        return fn

    return deco


def get(kind: str, name: str) -> Callable:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY[kind])) or "(none)"
        raise KeyError(f"no {kind} named {name!r}; known: {known}") from None


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY[kind])


# Import baseline components so they self-register.
from . import attention, mlp, norm, optim  # noqa: E402, F401
