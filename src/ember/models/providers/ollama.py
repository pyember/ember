"""Ollama provider for local completions."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.base import BaseProvider
from ember.models.schemas import ChatResponse, UsageStats


class OllamaProvider(BaseProvider):
    """Keyless provider that forwards requests to the local Ollama runtime."""

    requires_api_key: bool = False

    def _get_api_key_from_env(self) -> str:
        return ""

    @staticmethod
    def _base_url(**kwargs: Any) -> str:
        if "base_url" in kwargs:
            candidate = kwargs["base_url"]
            if not isinstance(candidate, str):
                raise ProviderAPIError(
                    "base_url must be a string",
                    context={"provider": "ollama", "value_type": type(candidate).__name__},
                )
            base_url = candidate.strip()
            return base_url or "http://localhost:11434"

        candidate = _config_value("base_url")
        if candidate is None:
            return "http://localhost:11434"
        if not isinstance(candidate, str):
            type_name = type(candidate).__name__
            raise TypeError(f"providers.ollama.base_url must be a string, got {type_name}")
        base_url = candidate.strip()
        return base_url or "http://localhost:11434"

    @staticmethod
    def _timeout(**kwargs: Any) -> float:
        if "timeout" in kwargs:
            try:
                return float(kwargs["timeout"])
            except (TypeError, ValueError) as exc:
                raise ProviderAPIError(
                    "timeout must be numeric (seconds)",
                    context={"provider": "ollama", "value": kwargs["timeout"]},
                ) from exc

        config_ms = _config_value("timeout_ms")
        if config_ms is None:
            return 30.0
        if isinstance(config_ms, bool):
            raise TypeError("providers.ollama.timeout_ms must be numeric")
        if isinstance(config_ms, (int, float)):
            return float(config_ms) / 1000.0
        if isinstance(config_ms, str):
            try:
                return float(config_ms) / 1000.0
            except ValueError as exc:
                raise TypeError("providers.ollama.timeout_ms must be numeric") from exc
        type_name = type(config_ms).__name__
        raise TypeError(f"providers.ollama.timeout_ms must be numeric, got {type_name}")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        raw_ratio = _config_value("token_char_ratio")
        if raw_ratio is None:
            ratio = 4.0
        elif isinstance(raw_ratio, bool):
            raise TypeError("providers.ollama.token_char_ratio must be numeric")
        elif isinstance(raw_ratio, (int, float)):
            ratio = float(raw_ratio)
        elif isinstance(raw_ratio, str):
            try:
                ratio = float(raw_ratio)
            except ValueError as exc:
                raise TypeError("providers.ollama.token_char_ratio must be numeric") from exc
        else:
            raise TypeError(
                f"providers.ollama.token_char_ratio must be numeric, got {type(raw_ratio).__name__}"
            )

        ratio = max(1.0, ratio)
        return max(1, int((len(text) / ratio) + 0.5))

    @staticmethod
    def _autopull_enabled(**kwargs: Any) -> bool:
        if "autopull" in kwargs:
            value = kwargs["autopull"]
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes"}
            raise ProviderAPIError(
                "autopull must be a boolean",
                context={"provider": "ollama", "value_type": type(value).__name__},
            )
        setting = _config_value("autopull")
        if isinstance(setting, str):
            return setting.lower() in {"1", "true", "yes"}
        if isinstance(setting, bool):
            return setting
        if setting is None:
            return False
        type_name = type(setting).__name__
        raise TypeError(f"providers.ollama.autopull must be a boolean, got {type_name}")

    def _model_exists(self, base_url: str, model: str, timeout: float) -> bool:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [
                    m.get("name") for m in (data or {}).get("models", []) if isinstance(m, dict)
                ]
                return model in set(models)
        except Exception:
            # If check fails, assume unknown to trigger standard flow
            return False

    def _choose_default_model(self, base_url: str, timeout: float) -> str:
        preferred = [
            "llama3.2:1b",
            "qwen2.5:0.5b",
            "tinyllama",
            "llama3.1:8b",
        ]
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(f"{base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                names = [
                    m.get("name") for m in (data or {}).get("models", []) if isinstance(m, dict)
                ]
                for p in preferred:
                    if p in names:
                        return p
                if names:
                    return names[0]
        except Exception:
            pass
        default_model = _config_value("default_model")
        if default_model is None:
            return "llama3.2:1b"
        if not isinstance(default_model, str):
            type_name = type(default_model).__name__
            raise TypeError(f"providers.ollama.default_model must be a string, got {type_name}")
        candidate = default_model.strip()
        return candidate or "llama3.2:1b"

    def _pull_model(
        self, base_url: str, model: str, timeout: float, progress: Any | None = None
    ) -> bool:
        """Pull a model via Ollama /api/pull; returns True on success."""
        try:
            with httpx.Client(timeout=timeout) as client:
                body = {"model": model, "stream": True}
                with client.stream("POST", f"{base_url}/api/pull", json=body) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            part = json.loads(line)
                        except Exception:
                            continue
                        if progress and callable(progress):  # user-supplied progress callback
                            try:
                                progress(part)
                            except Exception:
                                pass
                        status = str(part.get("status", "")).lower()
                        if status in {"success", "ok", "done"}:
                            return True
                        # Sometimes completion is indicated by completed == total
                        completed = part.get("completed")
                        total = part.get("total")
                        if (
                            isinstance(completed, int)
                            and isinstance(total, int)
                            and total > 0
                            and completed >= total
                        ):
                            return True
            return False
        except Exception:
            return False

    def complete(self, prompt: str, model: str, **kwargs: Any) -> ChatResponse:
        base_url = self._base_url(**kwargs)
        timeout = self._timeout(**kwargs)

        requested_model = model
        if model in {"auto", "default"}:
            requested_model = self._choose_default_model(base_url, timeout)

        options: dict[str, Any] = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs.pop("temperature")
        if "top_p" in kwargs:
            options["top_p"] = kwargs.pop("top_p")
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs.pop("max_tokens")
        if "stop" in kwargs:
            options["stop"] = kwargs.pop("stop")

        stream = bool(kwargs.pop("stream", False))

        system = kwargs.pop("context", None) or kwargs.pop("system", None)

        if system:
            url = f"{base_url}/api/chat"
            body: dict[str, Any] = {
                "model": requested_model,
                "messages": [
                    {"role": "system", "content": str(system)},
                    {"role": "user", "content": str(prompt)},
                ],
                "options": options,
                "stream": stream,
            }
        else:
            url = f"{base_url}/api/generate"
            body = {
                "model": requested_model,
                "prompt": str(prompt),
                "options": options,
                "stream": stream,
            }

        try:
            if self._autopull_enabled(**kwargs) and not self._model_exists(
                base_url, requested_model, timeout
            ):
                self._pull_model(base_url, requested_model, timeout, kwargs.get("progress"))
            if stream:
                gen = self.stream_complete(
                    prompt,
                    requested_model,
                    base_url=base_url,
                    timeout=timeout,
                    options=options,
                    system=system,
                )
                chunks: list[str] = []
                try:
                    while True:
                        chunks.append(next(gen))
                except StopIteration as exc:
                    final = exc.value
                    if not isinstance(final, ChatResponse):
                        raise ProviderAPIError(
                            "Streaming response was not finalized",
                            context={"provider": "ollama", "model": requested_model},
                        ) from exc
                    if not final.data:
                        final.data = "".join(chunks)
                    return final
            else:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=body)
                    resp.raise_for_status()
                    data = resp.json()
                    if system:
                        msg = data.get("message") or {}
                        text = msg.get("content", "") if isinstance(msg, dict) else ""
                    else:
                        text = data.get("response", "")
                    raw = data

                pt = self._estimate_tokens(str(prompt))
                ct = self._estimate_tokens(text)
                usage = UsageStats(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)

                return ChatResponse(
                    data=text or "", usage=usage, model_id=requested_model, raw_output=raw
                )

        except httpx.ConnectError as e:
            raise ProviderAPIError(
                "Could not reach Ollama server",
                context={
                    "model": model,
                    "error_type": "connection_refused",
                    "hint": "Run `ollama serve` or set providers.ollama.base_url",
                    "base_url": base_url,
                },
            ) from e
        except httpx.HTTPStatusError as e:
            detail = e.response.text if e.response is not None else str(e)
            # Common case: model not found (404) when not pulled yet
            hint = None
            if e.response is not None and e.response.status_code == 404:
                # Attempt autopull on failure if enabled, then retry once
                if self._autopull_enabled(**kwargs):
                    pulled = self._pull_model(
                        base_url, requested_model, timeout, kwargs.get("progress")
                    )
                    if pulled:
                        # Retry once after successful pull
                        return self.complete(prompt, requested_model, **kwargs)
                hint = f"Model '{requested_model}' not found. Try: ollama run {requested_model}"
            raise ProviderAPIError(
                f"Ollama HTTP error: {e}",
                context={"model": model, "detail": detail, "base_url": base_url, "hint": hint},
            ) from e
        except httpx.RequestError as e:
            raise ProviderAPIError(
                f"Ollama request failed: {e}", context={"model": model, "base_url": base_url}
            ) from e
        except Exception as e:
            raise ProviderAPIError(
                f"Unexpected Ollama error: {e}", context={"model": model, "base_url": base_url}
            ) from e

    def stream_complete(
        self, prompt: str, model: str, **kwargs: Any
    ) -> Generator[str, None, ChatResponse]:
        base_url = self._base_url(**kwargs)
        timeout = self._timeout(**kwargs)
        options = kwargs.get("options") or {}
        if not isinstance(options, dict):
            raise ProviderAPIError(
                "options must be a mapping",
                context={"provider": "ollama", "value_type": type(options).__name__},
            )
        system = kwargs.get("system")

        if model in {"auto", "default"}:
            model = self._choose_default_model(base_url, timeout)

        if system:
            url = f"{base_url}/api/chat"
            body: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": str(system)},
                    {"role": "user", "content": str(prompt)},
                ],
                "options": options,
                "stream": True,
            }
        else:
            url = f"{base_url}/api/generate"
            body = {
                "model": model,
                "prompt": str(prompt),
                "options": options,
                "stream": True,
            }

        text_accum: list[str] = []
        try:
            if self._autopull_enabled(**kwargs) and not self._model_exists(
                base_url, model, timeout
            ):
                self._pull_model(base_url, model, timeout, kwargs.get("progress"))
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            part = json.loads(line)
                        except Exception:
                            continue
                        if system:
                            msg = part.get("message") or {}
                            content = msg.get("content", "") if isinstance(msg, dict) else ""
                            if content:
                                text_accum.append(content)
                                yield content
                        else:
                            delta = part.get("response", "")
                            if delta:
                                text_accum.append(delta)
                                yield delta

            # Finalize
            text = "".join(text_accum)
            pt = self._estimate_tokens(str(prompt))
            ct = self._estimate_tokens(text)
            usage = UsageStats(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
            final = ChatResponse(
                data=text,
                usage=usage,
                model_id=model,
                raw_output={"streamed": True, "endpoint": url},
            )
            return final
        except httpx.ConnectError as e:
            raise ProviderAPIError(
                "Could not reach Ollama server",
                context={
                    "model": model,
                    "error_type": "connection_refused",
                    "hint": "Run `ollama serve` or set providers.ollama.base_url",
                    "base_url": base_url,
                },
            ) from e
        except httpx.RequestError as e:
            raise ProviderAPIError(
                f"Ollama request failed: {e}", context={"model": model, "base_url": base_url}
            ) from e


def _config_value(field: str) -> object | None:
    from ember._internal.context.runtime import EmberContext

    return EmberContext.current().get_config(f"providers.ollama.{field}")
