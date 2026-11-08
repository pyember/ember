"""A stand-in for `ember.models` until the models lane merges.

This module is deleted when the operators lane rebases onto it; nothing else
in the package changes except the import that points here.
"""

from collections.abc import Sequence

from ember.errors import ConfigError
from ember.llm.completion import Completion
from ember.llm.ref import ModelRef
from ember.llm.request import Message

__all__ = ["models"]


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
