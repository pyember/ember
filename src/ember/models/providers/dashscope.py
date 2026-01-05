"""DashScope provider for Alibaba Qwen models."""

from __future__ import annotations

from typing import Any

import httpx

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats


class DashScopeProvider(BaseProvider):
    """DashScope/Qwen provider that adapts responses into Ember outputs."""

    BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    def _get_api_key_from_env(self) -> str:
        from ember._internal.context.runtime import EmberContext

        return EmberContext.current().get_credential("dashscope")

    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        stream_requested = bool(kwargs.pop("stream", False))
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "enable" if stream_requested else "disable",
        }

        system = kwargs.pop("context", None) or kwargs.pop("system", None)
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": prompt})

        parameters: dict[str, object] = {}
        extra_body: dict[str, object] = {}

        temperature = kwargs.pop("temperature", None)
        if temperature is not None:
            parameters["temperature"] = temperature

        top_p = kwargs.pop("top_p", None)
        if top_p is not None:
            parameters["top_p"] = top_p

        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens

        thinking = bool(kwargs.pop("thinking", False))
        if thinking:
            extra_body["enable_thinking"] = True
            thinking_budget = kwargs.pop("thinking_budget", None)
            if thinking_budget is not None:
                try:
                    extra_body["thinking_budget"] = int(thinking_budget)
                except (TypeError, ValueError) as exc:
                    raise ProviderAPIError(
                        "DashScope thinking_budget must be an int",
                        context={"model": model, "value": thinking_budget},
                    ) from exc

        if kwargs:
            parameters.update(kwargs)

        body: dict[str, object] = {
            "model": model,
            "input": {"messages": messages},
            "parameters": parameters,
        }
        if extra_body:
            body["extra_body"] = extra_body

        try:
            with httpx.Client(timeout=60) as client:
                response = client.post(self.BASE_URL, headers=headers, json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            raise ProviderAPIError(
                f"DashScope HTTP error: {exc}",
                context={"model": model, "detail": detail},
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderAPIError(
                f"DashScope request failed: {exc}",
                context={"model": model},
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderAPIError(
                "DashScope returned invalid JSON",
                context={"model": model},
            ) from exc

        text = ""
        if isinstance(data, dict):
            output = data.get("output")
            if isinstance(output, dict):
                text = str(output.get("text") or "")

        usage = None
        raw_usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(raw_usage, dict):
            prompt_tokens = int(raw_usage.get("input_tokens") or 0)
            completion_tokens = int(raw_usage.get("output_tokens") or 0)
            usage = UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        return ChatResponse(data=text, usage=usage, model_id=model, raw_output=data)
