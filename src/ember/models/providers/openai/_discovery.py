from __future__ import annotations

from collections.abc import Callable, Iterable

import openai

from ember.models.discovery.provider_api import DiscoveryProvider
from ember.models.discovery.types import DiscoveredModel

from ._client import resolve_openai_api_key


class OpenAIDiscoveryAdapter(DiscoveryProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key_resolver: Callable[[], str | None] | None = None,
        client_factory: Callable[[str], openai.OpenAI] | None = None,
    ) -> None:
        self._resolve_api_key = api_key_resolver or resolve_openai_api_key
        self._client_factory = client_factory or (lambda key: openai.OpenAI(api_key=key))

    def list_models(
        self,
        *,
        region: str | None = None,
        project_hint: str | None = None,
    ) -> Iterable[DiscoveredModel]:
        api_key = self._resolve_api_key()
        if not api_key:
            return []

        client = self._client_factory(api_key)
        response = client.models.list()  # type: ignore[call-arg]
        return [
            DiscoveredModel(
                provider="openai",
                id=entry.id,
                display_name=getattr(entry, "display_name", None)
                or getattr(entry, "displayName", None),
                raw_payload=None,
            )
            for entry in response.data
        ]
