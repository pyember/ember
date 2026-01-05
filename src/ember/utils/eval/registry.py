"""Evaluator registry utilities."""

from __future__ import annotations

from collections.abc import Callable

from .base_evaluator import IEvaluator


class EvaluatorRegistry:
    """Registry for evaluator factories."""

    def __init__(self) -> None:
        self._registry: dict[str, Callable[..., IEvaluator]] = {}

    def register(self, name: str, factory: Callable[..., IEvaluator]) -> None:
        self._registry[name] = factory

    def create(self, name: str, **kwargs: object) -> IEvaluator:
        try:
            factory = self._registry[name]
        except KeyError as exc:
            raise KeyError(f"No evaluator registered with name: {name}") from exc
        return factory(**kwargs)

    def names(self) -> list[str]:
        return list(self._registry.keys())
