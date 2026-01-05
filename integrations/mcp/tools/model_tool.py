"""Lightweight wrappers for invoking Ember model bindings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from ember.api import models as models_api
from ember.api.models import ModelBinding, Response


@dataclass(slots=True)
class InvokeResult:
    """Normalized result returned by :class:`ModelTool`."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model_id: Optional[str]
    response: Response

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class ModelTool:
    """Thin async facade over a :class:`ModelBinding`."""

    name: str
    binding: ModelBinding
    default_params: Mapping[str, Any] | None = None

    async def invoke(
        self,
        *,
        prompt: str,
        system: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> InvokeResult:
        """Run the bound model in a worker thread and normalize the result."""

        merged: Dict[str, Any] = {}
        if self.default_params:
            merged.update(self.default_params)
        if params:
            merged.update(params)
        if system is not None:
            merged.setdefault("system", system)

        response = await asyncio.to_thread(self.binding.response, prompt, **merged)
        usage = response.usage

        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cost_usd = float(usage.get("cost", 0.0) or 0.0)

        return InvokeResult(
            text=response.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            model_id=response.model_id,
            response=response,
        )


class ModelToolRegistry:
    """Registry of model tools addressable by MCP clients."""

    def __init__(self, bindings: Mapping[str, ModelBinding] | None = None):
        self._bindings: Dict[str, ModelBinding] = dict(bindings or {})
        self._tools: Dict[str, ModelTool] = {}

    @classmethod
    def from_model_ids(
        cls,
        model_ids: Sequence[str],
        *,
        default_params: Mapping[str, Any] | None = None,
    ) -> "ModelToolRegistry":
        """Instantiate bindings for the provided model ids."""

        registry = cls()
        for model_id in model_ids:
            binding = models_api.instance(model_id)
            registry.register(model_id, binding, default_params=default_params)
        return registry

    def register(
        self,
        model_id: str,
        binding: ModelBinding,
        *,
        default_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Register or override a model binding."""

        self._bindings[model_id] = binding
        self._tools.pop(model_id, None)
        if default_params:
            self._tools[model_id] = ModelTool(model_id, binding, default_params=default_params)

    def has(self, model_id: str) -> bool:
        """Return ``True`` if a tool is registered for ``model_id``."""

        return model_id in self._bindings

    def get(
        self,
        model_id: str,
        *,
        default_params: Mapping[str, Any] | None = None,
    ) -> ModelTool:
        """Return a lazily constructed :class:`ModelTool`."""

        if model_id not in self._bindings:
            raise KeyError(model_id)
        if model_id not in self._tools:
            binding = self._bindings[model_id]
            params = default_params or {}
            self._tools[model_id] = ModelTool(model_id, binding, default_params=params)
        return self._tools[model_id]

    def ensure_tool(self, model_id: str) -> ModelTool:
        """Convenience wrapper for :meth:`get` with default parameters."""

        return self.get(model_id)

