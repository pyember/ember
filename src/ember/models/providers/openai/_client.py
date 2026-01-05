from __future__ import annotations

import openai


def resolve_openai_api_key() -> str:
    """Resolve OpenAI API key from credential store.

    Returns:
        The OpenAI API key string.

    Raises:
        ValueError: If the OpenAI credential is not configured.
    """
    from ember._internal.context.runtime import EmberContext
    from ember.core.credentials import CredentialNotFoundError

    try:
        return EmberContext.current().get_credential("openai")
    except CredentialNotFoundError as exc:
        raise ValueError(
            "OpenAI credential not configured. "
            "Set OPENAI_API_KEY or configure via ~/.ember/config.yaml"
        ) from exc


def create_openai_client(api_key: str) -> openai.OpenAI:
    from ember._internal.context.runtime import EmberContext

    if not api_key or not api_key.strip():
        raise ValueError("OpenAI api_key must be a non-empty string")

    client_kwargs: dict[str, object] = {"api_key": api_key}

    base_url = EmberContext.current().get_config("providers.openai.base_url")
    if isinstance(base_url, str) and base_url.strip():
        client_kwargs["base_url"] = base_url.strip()

    # Disable SDK-level retry. The SDK's default (max_retries=2) uses very short
    # delays (0.5-1s) that cause thundering herd on rate limits. We rely on
    # tenacity at the provider layer for proper exponential backoff.
    client_kwargs["max_retries"] = 0

    return openai.OpenAI(**client_kwargs)
