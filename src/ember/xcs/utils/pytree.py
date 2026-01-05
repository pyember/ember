"""JAX pytree helpers used across XCS internals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

try:
    import jax.tree_util as _tree_util
except Exception:  # pragma: no cover - optional dependency
    _tree_util = None

from ember._internal.module import Module


@dataclass(slots=True)
class StaticWrapper:
    """Wraps arbitrary Python objects so JAX treats them as static leaves."""

    value: Any

    def tree_flatten(self) -> Tuple[Iterable[Any], Any]:
        """Return the pytree flatten representation."""
        return [], self.value

    @classmethod
    def tree_unflatten(cls, static: Any, children: Iterable[Any]) -> "StaticWrapper":
        """Reconstruct the wrapper from pytree pieces."""
        del children  # Static wrappers never have dynamic children.
        return cls(static)


def register_ember_pytrees() -> None:
    """Register Ember modules and wrappers with JAX tree utilities."""
    if _tree_util is None:
        return

    if getattr(register_ember_pytrees, "_registered", False):
        return

    _tree_util.register_pytree_node(
        StaticWrapper, StaticWrapper.tree_flatten, StaticWrapper.tree_unflatten
    )

    register_ember_pytrees._registered = True  # type: ignore[attr-defined]


def ensure_pytree_compatible(value: Any) -> Any:
    """Ensure `value` can participate in JAX pytree transformations."""
    if isinstance(value, Module):
        return value

    if _tree_util is None:
        return StaticWrapper(value)

    try:
        _tree_util.tree_flatten(value)
        return value
    except (TypeError, ValueError):
        return StaticWrapper(value)


__all__ = ["StaticWrapper", "register_ember_pytrees", "ensure_pytree_compatible"]
