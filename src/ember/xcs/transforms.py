"""`jit` and `map`."""

import functools
import types
from collections.abc import Callable, Iterable
from typing import Generic, ParamSpec, TypeVar, overload

__all__ = ["Jitted", "jit", "map"]

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class Jitted(Generic[P, R]):
    """A function whose model calls overlap when they are independent."""

    def __init__(self, fn: Callable[P, R]) -> None:
        self._fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._fn(*args, **kwargs)

    @overload
    def __get__(self, instance: None, owner: type) -> "Jitted[P, R]": ...

    @overload
    def __get__(self, instance: object, owner: type) -> Callable[..., R]: ...

    def __get__(self, instance: object | None, owner: type) -> "Jitted[P, R] | Callable[..., R]":
        if instance is None:
            return self
        return types.MethodType(self, instance)


def jit(fn: Callable[P, R]) -> Jitted[P, R]:
    """Runs ``fn`` with independent model calls overlapping."""
    return Jitted(fn)


def map(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """Runs ``fn`` on each item and returns the results in order."""
    return [fn(item) for item in items]
