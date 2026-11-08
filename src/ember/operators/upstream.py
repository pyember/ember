"""Stand-ins for `ember.models` and `ember.xcs.map` until those lanes merge.

This module is deleted when the operators lane rebases onto them; nothing
else in the package changes except the two imports that point here.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from ember.errors import ConfigError
from ember.llm.completion import Completion
from ember.llm.ref import ModelRef
from ember.llm.request import Message

__all__ = ["models", "xcs_map"]

T = TypeVar("T")
R = TypeVar("R")


def models(
    model: str | ModelRef,
    prompt: str | Sequence[Message],
    *,
    system: str | None,
    temperature: float | None,
    top_p: float | None,
    max_output_tokens: int | None,
    stop: Sequence[str],
    seed: int | None,
    reasoning_effort: str | None,
) -> Completion:
    """The documented `ember.models` call, restricted to what operators pass."""
    del prompt, system, temperature, top_p, max_output_tokens, stop, seed, reasoning_effort
    raise ConfigError(f"ember.models is not on this branch yet; cannot call {model}")


def xcs_map(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """The documented `ember.xcs.map`, run sequentially."""
    return [fn(item) for item in items]
