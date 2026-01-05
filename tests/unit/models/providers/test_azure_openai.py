"""Unit tests for the AzureOpenAIProvider."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from ember.models.providers.azure_openai import (
    AzureOpenAIProvider,
    create_azure_shard_class,
)

pytestmark = pytest.mark.usefixtures("tmp_ctx")


@patch("openai.OpenAI")
def test_shard_factory_overrides_endpoint(mock_openai, tmp_ctx):
    """Shard factory binds endpoint/api-version for each instance."""
    shard_cls = create_azure_shard_class(
        "eastus2",
        endpoint="https://eastus2.azure.openai",
        api_version="2024-10-21",
    )

    shard_cls(api_key="test-key")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://eastus2.azure.openai/openai/deployments",
        default_query={"api-version": "2024-10-21"},
    )


@patch("openai.OpenAI")
def test_responses_path_for_configured_deployment(mock_openai, tmp_ctx):
    """Deployments listed in responses_deployments use the Responses API."""
    tmp_ctx.set_config("providers.azure_openai.endpoint", "https://azure.local")
    tmp_ctx.set_config("providers.azure_openai.responses_deployments", ["gpt5-prod"])

    mock_client = Mock()
    mock_openai.return_value = mock_client
    mock_client.chat = SimpleNamespace(completions=Mock())
    mock_client.chat.completions.create.return_value = Mock()

    responses_client = Mock()
    responses_client.create.return_value = SimpleNamespace(
        output_text="hi there",
        usage=SimpleNamespace(input_tokens=4, output_tokens=6, total_tokens=10),
    )
    mock_client.responses = responses_client

    provider = AzureOpenAIProvider(api_key="test-key")
    result = provider.complete("ping", "gpt5-prod")

    responses_client.create.assert_called_once()
    assert mock_client.chat.completions.create.call_count == 0

    kwargs = responses_client.create.call_args.kwargs
    assert kwargs["model"] == "gpt5-prod"
    assert kwargs["input"][-1]["content"][0]["text"] == "ping"

    assert result.data == "hi there"
    assert result.usage.total_tokens == 10
    assert result.model_id == "gpt5-prod"


@patch("openai.OpenAI")
def test_chat_path_for_standard_deployment(mock_openai, tmp_ctx):
    """Default deployments keep using Chat Completions."""
    tmp_ctx.set_config("providers.azure_openai.endpoint", "https://azure.local")

    mock_client = Mock()
    mock_openai.return_value = mock_client

    chat_response = Mock()
    chat_response.choices = [Mock(message=Mock(content="pong"))]
    chat_response.usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5, total_tokens=8)

    mock_client.chat = SimpleNamespace(completions=Mock())
    mock_client.chat.completions.create.return_value = chat_response
    mock_client.responses = Mock()

    provider = AzureOpenAIProvider(api_key="test-key")
    result = provider.complete("ping", "prod-4o", temperature=0.2)

    mock_client.chat.completions.create.assert_called_once()
    assert result.data == "pong"
    assert result.usage.total_tokens == 8
    assert result.model_id == "prod-4o"
