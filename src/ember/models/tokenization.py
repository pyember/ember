"""Provider-aware token counting utilities.

The helpers below prefer native provider tooling when available and fall back to
``tiktoken`` to preserve deterministic estimates. Callers should treat failures
as actionable configuration or dependency errors.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import tiktoken

from ember.core.credentials import CredentialNotFoundError
from ember.models.catalog import get_model_info

_OPENAI_MODEL_ALIASES = {
    "gpt-5": "cl100k_base",
    "gpt-5-mini": "cl100k_base",
    "gpt-5-nano": "cl100k_base",
    "gpt-5-chat-latest": "cl100k_base",
    "gpt-5.1": "cl100k_base",
    "gpt-5.1-codex": "cl100k_base",
    "gpt-5.1-codex-max": "cl100k_base",
    "gpt-5.1-chat-latest": "cl100k_base",
    "gpt-5.2": "cl100k_base",
    "gpt-5.2-chat-latest": "cl100k_base",
    "gpt-5.2-pro": "cl100k_base",
    "gpt-5-codex": "cl100k_base",
    "gpt-5-pro": "cl100k_base",
}


@lru_cache(maxsize=256)
def _openai_encoding(model_id: str) -> tiktoken.Encoding:
    """Return the tokenizer encoding for an OpenAI model.

    Args:
        model_id: Canonical OpenAI model identifier.

    Returns:
        The matching ``tiktoken`` encoding.

    Raises:
        ValueError: If ``model_id`` is not supported by ``tiktoken``.
    """

    encoded_id = _OPENAI_MODEL_ALIASES.get(model_id, model_id)

    try:
        return tiktoken.encoding_for_model(encoded_id)
    except KeyError:
        if encoded_id == "cl100k_base":
            return tiktoken.get_encoding("cl100k_base")

    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unsupported OpenAI model id for tokenization: {model_id}") from exc


def _openai_count(text: str, model_id: str) -> int:
    """Count tokens for OpenAI models using tiktoken.

    Args:
        text: Prompt text to tokenize.
        model_id: Canonical OpenAI model identifier.

    Returns:
        Number of tokens according to the model's tokenizer.
    """

    encoding = _openai_encoding(model_id)
    return len(encoding.encode(text))


def _anthropic_count(text: str, model_id: str) -> int:
    """Count tokens using Anthropic's official API.

    Args:
        text: Prompt text to tokenize.
        model_id: Canonical Anthropic model identifier.

    Returns:
        Number of tokens reported by Anthropic.

    Raises:
        RuntimeError: If an API key is unavailable.
    """

    try:
        from anthropic import Anthropic
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install the 'anthropic' package to count tokens for Claude models"
        ) from exc

    from ember.models.providers.anthropic import _resolve_anthropic_api_key

    try:
        api_key = _resolve_anthropic_api_key()
    except CredentialNotFoundError as exc:
        raise RuntimeError(
            "Missing Anthropic credential for token counting. "
            "Set providers.anthropic.api_key (use `ember configure` or `ember setup`)."
        ) from exc

    client = Anthropic(api_key=api_key)
    response = client.messages.count_tokens(
        model=model_id,
        messages=[{"role": "user", "content": text}],
    )
    return int(response.input_tokens)


def _google_count(text: str, model_id: str) -> int:
    """Count tokens using Google's Generative AI SDK.

    Args:
        text: Prompt text to tokenize.
        model_id: Canonical Gemini model identifier.

    Returns:
        Number of tokens reported by Google.
    """

    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install 'google-generativeai' to count tokens for Gemini models"
        ) from exc

    try:
        from ember._internal.context.runtime import EmberContext

        api_key = EmberContext.current().get_credential("google")
    except CredentialNotFoundError as exc:
        raise RuntimeError(
            "Missing Google credential for token counting. "
            "Set providers.google.api_key (use `ember configure` or `ember setup`)."
        ) from exc

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(model_id)
    response = model.count_tokens(text)
    return int(response.total_tokens)


def count_tokens(text: str, model_id: str, provider: Optional[str] = None) -> int:
    """Count tokens for ``text`` using the configured provider tokenizer.

    Args:
        text: Prompt text to tokenize.
        model_id: Canonical model identifier registered with Ember.
        provider: Optional provider slug override; defaults to catalog metadata.

    Returns:
        Number of tokens consumed by ``text``.

    Raises:
        ValueError: If the provider is unsupported or the model id is unknown.
        RuntimeError: If required provider credentials are missing.
    """

    provider_name = provider
    if provider_name is None:
        try:
            provider_name = get_model_info(model_id).provider
        except KeyError as exc:  # pragma: no cover
            raise ValueError(f"Unknown model id: {model_id}") from exc

    if provider_name == "openai":
        return _openai_count(text, model_id)

    if provider_name == "anthropic":
        return _anthropic_count(text, model_id)

    if provider_name == "google":
        return _google_count(text, model_id)

    raise ValueError(f"Unsupported provider '{provider_name}' for token counting")


__all__ = ["count_tokens"]
