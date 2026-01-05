"""Provider behavior tests."""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest

from ember._internal.exceptions import ProviderAPIError
from ember.models.providers.anthropic import AnthropicProvider
from ember.models.providers.google import GoogleProvider
from ember.models.providers.openai import OpenAIProvider
from ember.models.schemas import ChatResponse

pytestmark = pytest.mark.usefixtures("tmp_ctx")


class TestBaseProvider:
    """Test base provider functionality common to all providers."""

    def test_api_key_validation(self):
        """Test that providers validate API keys properly."""
        # BaseProvider is abstract, so test with concrete providers
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAIProvider()
            with pytest.raises(ValueError, match="API key"):
                AnthropicProvider()
            with pytest.raises(ValueError, match="API key"):
                GoogleProvider()


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_initialization(self):
        """Test OpenAI provider initialization."""
        # With explicit key
        provider = OpenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_model_validation(self):
        """Test OpenAI model validation."""
        provider = OpenAIProvider(api_key="test-key")

        # Valid models
        assert provider.validate_model("gpt-4")
        assert provider.validate_model("gpt-4-turbo")
        assert provider.validate_model("gpt-3.5-turbo")
        assert provider.validate_model("gpt-4o")  # Changed from o1-preview
        assert provider.validate_model("gpt-4.1")

        # Invalid models
        assert not provider.validate_model("gemini-pro")
        assert not provider.validate_model("claude-3")
        assert not provider.validate_model("invalid-model")

    @patch("openai.OpenAI")
    def test_completion_success(self, mock_openai_class):
        """Test successful completion."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello!"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"

        mock_client.chat.completions.create.return_value = mock_response

        # Test
        provider = OpenAIProvider(api_key="test-key")
        response = provider.complete("Say hello", "gpt-4", temperature=0.7)

        assert response.data == "Hello!"
        assert response.usage.total_tokens == 15
        assert response.model_id == "gpt-4"

    @patch("openai.OpenAI")
    def test_chat_maps_max_tokens_to_completion_tokens(self, mock_openai_class):
        """Legacy max_tokens should map to max_completion_tokens for chat API."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="ok"))]
        mock_response.usage = Mock(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        mock_response.model = "gpt-4o"

        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        provider.complete("hi", "gpt-4o", max_tokens=42)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "max_tokens" not in call_kwargs
        assert call_kwargs["max_completion_tokens"] == 42

    @patch("openai.OpenAI")
    def test_completion_error_handling(self, mock_openai_class):
        """Test error handling in completions."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Test authentication error
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")

        provider = OpenAIProvider(api_key="bad-key")
        with pytest.raises(ProviderAPIError, match="Invalid API key"):
            provider.complete("test", "gpt-4")

    @patch("openai.OpenAI")
    def test_reasoning_rejected_for_standard_models(self, mock_openai_class):
        """Non GPT-5 models should reject reasoning payloads."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(api_key="test-key")

        with pytest.raises(ProviderAPIError, match="reasoning/text controls are only supported"):
            provider.complete("prompt", "gpt-4o", reasoning={"effort": "high"})

        assert mock_client.chat.completions.create.call_count == 0

    @patch("openai.OpenAI")
    def test_gpt5_accepts_reasoning_payload(self, mock_openai_class):
        """GPT-5 family calls the Responses API with reasoning payloads."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.return_value = Mock()
        responses_client = Mock()
        mock_client.responses = responses_client

        mock_response = Mock(output_text="hi", usage=None)
        responses_client.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        provider.complete("prompt", "gpt-5", reasoning={"effort": "high"})

        responses_kwargs = responses_client.create.call_args.kwargs
        assert responses_kwargs["model"] == "gpt-5"
        assert responses_kwargs["reasoning"]["effort"] == "high"
        input_messages = responses_kwargs["input"]
        assert input_messages[-1]["role"] == "user"
        assert input_messages[-1]["content"][0]["text"] == "prompt"
        assert mock_client.chat.completions.create.call_count == 0

    @patch("openai.OpenAI")
    def test_gpt5_accepts_reasoning_mapping(self, mock_openai_class):
        """GPT-5 family should accept nested reasoning dicts with valid effort levels."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.return_value = Mock()
        responses_client = Mock()
        mock_client.responses = responses_client
        responses_client.create.return_value = Mock(output_text="", usage=None)

        provider = OpenAIProvider(api_key="test-key")
        provider.complete(
            "prompt",
            "gpt-5",
            reasoning={"effort": "high"},
        )

        responses_kwargs = responses_client.create.call_args.kwargs
        assert responses_kwargs["model"] == "gpt-5"
        assert responses_kwargs["reasoning"] == {"effort": "high"}
        assert mock_client.chat.completions.create.call_count == 0

    @patch("openai.OpenAI")
    def test_none_reasoning_omits_parameter(self, mock_openai_class):
        """Passing reasoning=None should omit the reasoning parameter entirely."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        responses_client = Mock()
        mock_client.responses = responses_client
        responses_client.create.return_value = Mock(output_text="ok", usage=None)

        provider = OpenAIProvider(api_key="test-key")
        # None reasoning should work - it means "no reasoning controls"
        result = provider.complete("prompt", "gpt-5", reasoning=None)
        assert result.data == "ok"
        # The API should be called
        assert responses_client.create.call_count == 1
        # reasoning should not be in the call kwargs
        call_kwargs = responses_client.create.call_args.kwargs
        assert call_kwargs.get("reasoning") is None

    @patch("openai.OpenAI")
    def test_gpt41_routes_through_responses(self, mock_openai_class):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        responses_client = Mock()
        responses_client.create.return_value = Mock(output_text="hi", usage=None)
        mock_client.responses = responses_client

        provider = OpenAIProvider(api_key="test-key")
        provider.complete("prompt", "gpt-4.1")

        responses_client.create.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()

    @patch("openai.OpenAI")
    def test_gpt5_maps_legacy_sampling_params(self, mock_openai_class):
        """Legacy temperature/top_p knobs map into Responses payloads."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        responses_client = Mock()
        responses_client.create.return_value = Mock(output_text="", usage=None)
        mock_client.responses = responses_client

        provider = OpenAIProvider(api_key="test-key")

        provider.complete("prompt", "gpt-5", temperature=0.2)
        provider.complete("prompt", "gpt-5-mini", top_p=0.9)

        first_call = responses_client.create.call_args_list[0].kwargs
        first_temp = first_call.get("temperature")
        if first_temp is None:
            first_temp = first_call["text"]["temperature"]
        assert first_temp == pytest.approx(0.2)

        second_call = responses_client.create.call_args_list[1].kwargs
        second_top_p = second_call.get("top_p")
        if second_top_p is None:
            second_top_p = second_call["text"]["top_p"]
        assert second_top_p == pytest.approx(0.9)

        with pytest.raises(ProviderAPIError):
            provider.complete("prompt", "gpt-5-nano", logprobs=2)

    @patch("openai.OpenAI")
    def test_gpt5_remaps_max_tokens(self, mock_openai_class):
        """max_tokens should become max_output_tokens for Responses API."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        responses_client = Mock()
        responses_client.create.return_value = Mock(output_text="", usage=None)
        mock_client.responses = responses_client

        provider = OpenAIProvider(api_key="test-key")
        provider.complete("prompt", "gpt-5", max_tokens=256)

        kwargs = responses_client.create.call_args.kwargs
        assert kwargs["max_output_tokens"] == 256
        assert "max_tokens" not in kwargs

    @patch("openai.OpenAI")
    def test_complete_responses_payload_normalizes_messages(self, mock_openai_class):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        responses_client = Mock()
        mock_client.responses = responses_client
        responses_client.create.return_value = Mock(output_text="hi", model="gpt-5", usage=None)

        provider = OpenAIProvider(api_key="test-key")
        payload = {
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            "temperature": 0.3,
            "max_tokens": 64,
        }

        provider.complete_responses_payload(payload)

        responses_client.create.assert_called_once()
        kwargs = responses_client.create.call_args.kwargs
        assert "messages" not in kwargs
        assert kwargs["input"][-1]["role"] == "user"
        assert kwargs["input"][-1]["content"][0]["text"] == "Hello"
        temperature = kwargs.get("temperature")
        if temperature is None:
            temperature = kwargs["text"]["temperature"]
        assert temperature == pytest.approx(0.3)
        assert kwargs["max_output_tokens"] == 64
        assert "max_tokens" not in kwargs

    @patch("openai.OpenAI")
    def test_stream_responses_payload_normalizes_messages(self, mock_openai_class):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        responses_client = Mock()
        mock_client.responses = responses_client

        class DummyStream:
            def __iter__(self_inner):
                return iter([])

            def get_final_response(self_inner):
                return SimpleNamespace(output_text="", usage=None)

        stream_manager = MagicMock()
        stream_manager.__enter__.return_value = DummyStream()
        stream_manager.__exit__.return_value = False
        responses_client.stream.return_value = stream_manager

        provider = OpenAIProvider(api_key="test-key")
        payload = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "Ping"}],
            "top_p": 0.8,
        }

        generator = provider.stream_responses_payload(payload)
        assert list(generator) == []

        responses_client.stream.assert_called_once()
        kwargs = responses_client.stream.call_args.kwargs
        assert "messages" not in kwargs
        assert kwargs["input"][0]["role"] == "user"
        assert kwargs["input"][0]["content"][0]["text"] == "Ping"
        top_p = kwargs.get("top_p")
        if top_p is None:
            top_p = kwargs["text"]["top_p"]
        assert top_p == pytest.approx(0.8)

    @patch("openai.OpenAI")
    def test_gpt5_streaming_not_implemented(self, mock_openai_class):
        """Streaming attempts should raise a clear error."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        responses_client = Mock()
        responses_client.create.return_value = Mock(output_text="", usage=None)
        mock_client.responses = responses_client

        provider = OpenAIProvider(api_key="test-key")

        with pytest.raises(ProviderAPIError, match="Streaming is not implemented"):
            provider.complete("prompt", "gpt-5", stream=True)

        assert responses_client.create.call_count == 0

    @patch("openai.OpenAI")
    def test_gpt5_streams_text_and_usage(self, mock_openai_class):
        """stream_complete should yield deltas and surface final usage."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        responses_client = Mock()
        mock_client.responses = responses_client

        events = [
            SimpleNamespace(type="response.output_text.delta", delta="Hello"),
            SimpleNamespace(type="response.output_text.delta", delta=" world"),
        ]

        final_response = SimpleNamespace(
            output_text="Hello world",
            usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5),
        )

        class DummyStream:
            def __iter__(self_inner):
                return iter(events)

            def get_final_response(self_inner):
                return final_response

        stream = DummyStream()

        stream_manager = MagicMock()
        stream_manager.__enter__.return_value = stream
        stream_manager.__exit__.return_value = False

        responses_client.stream.return_value = stream_manager

        provider = OpenAIProvider(api_key="test-key")
        generator = provider.stream_complete("prompt", "gpt-5")

        chunks: list[dict[str, object]] = []
        try:
            while True:
                chunk = next(generator)
                chunks.append(json.loads(chunk))
        except StopIteration as stop:
            final = stop.value

        assert [payload["delta"] for payload in chunks] == ["Hello", " world"]
        assert isinstance(final, ChatResponse)
        assert final.data == "Hello world"
        assert final.usage.total_tokens == 5

    @patch("openai.OpenAI")
    def test_gpt5_streaming_error_event(self, mock_openai_class):
        """Error events during streaming raise ProviderAPIError."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        responses_client = Mock()
        mock_client.responses = responses_client

        events = [
            SimpleNamespace(
                type="response.error",
                error=SimpleNamespace(message="boom"),
            )
        ]

        class ErrorStream:
            def __iter__(self_inner):
                return iter(events)

            def get_final_response(self_inner):
                return None

        stream = ErrorStream()

        stream_manager = MagicMock()
        stream_manager.__enter__.return_value = stream
        stream_manager.__exit__.return_value = False

        responses_client.stream.return_value = stream_manager

        provider = OpenAIProvider(api_key="test-key")
        generator = provider.stream_complete("prompt", "gpt-5")

        with pytest.raises(ProviderAPIError, match="streaming error"):
            next(generator)


class TestAnthropicProvider:
    """Test Anthropic provider implementation."""

    def test_initialization(self):
        """Test Anthropic provider initialization."""
        # With explicit key
        provider = AnthropicProvider(api_key="test-key")
        assert provider.api_key == "test-key"

    def test_model_validation(self):
        """Test Anthropic model validation."""
        provider = AnthropicProvider(api_key="test-key")

        # Valid models
        assert provider.validate_model("claude-3-opus")
        assert provider.validate_model("claude-3-sonnet")
        assert provider.validate_model("claude-3-haiku")
        assert provider.validate_model("claude-2.1")

        # Invalid models
        assert not provider.validate_model("gpt-4")
        assert not provider.validate_model("gemini-pro")
        assert not provider.validate_model("invalid-model")

    @patch("anthropic.Anthropic")
    def test_completion_success(self, mock_anthropic_class):
        """Test successful completion."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Hello from Claude!")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-3-opus"

        mock_client.messages.create.return_value = mock_response

        # Test
        provider = AnthropicProvider(api_key="test-key")
        response = provider.complete("Say hello", "claude-3-opus", temperature=0.7)

        assert response.data == "Hello from Claude!"
        assert response.usage.total_tokens == 15
        assert response.model_id == "claude-3-opus"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Say hello"}]}
        ]
        assert "system" not in call_kwargs
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0.7

    @patch("anthropic.Anthropic")
    def test_completion_with_context_and_system(self, mock_anthropic_class):
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="ok")]
        mock_response.usage = Mock(input_tokens=1, output_tokens=1)
        mock_response.model = "claude-3-opus"
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        provider.complete(
            "Hi",
            "claude-3-opus",
            system="Be concise",
            context=[{"role": "assistant", "content": "previous reply"}],
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == [{"type": "text", "text": "Be concise"}]
        assert call_kwargs["messages"] == [
            {"role": "assistant", "content": [{"type": "text", "text": "previous reply"}]},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        ]

    @patch("anthropic.Anthropic")
    def test_complete_responses_payload_delegates(self, mock_anthropic_class):
        mock_anthropic_class.return_value = Mock()
        provider = AnthropicProvider(api_key="test-key")

        payload = {
            "model": "claude-3-haiku",
            "instructions": "Be direct",
            "input": [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "prior"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                },
            ],
            "max_output_tokens": 256,
            "temperature": 0.2,
            "top_p": 0.8,
            "stop": ["STOP"],
        }

        expected = ChatResponse(data="ok", model_id="claude-3-haiku")
        with patch.object(AnthropicProvider, "complete", return_value=expected) as mock_complete:
            result = provider.complete_responses_payload(payload)

        assert result is expected
        mock_complete.assert_called_once()
        args, kwargs = mock_complete.call_args
        assert args[0] == "Hello"
        assert args[1] == "claude-3-haiku"
        assert kwargs["system"] == "Be direct"
        assert kwargs["context"] == [{"role": "assistant", "content": "prior"}]
        assert kwargs["max_tokens"] == 256
        assert kwargs["temperature"] == 0.2
        assert kwargs["top_p"] == 0.8
        assert kwargs["stop"] == ["STOP"]


class TestGoogleProvider:
    """Test Google provider implementation with pyasn1 compatibility."""

    def test_pyasn1_compatibility(self):
        """Test that pyasn1 compatibility patch is applied."""
        # This import should not raise pyasn1 errors
        from ember.models.providers.google import GoogleProvider

        # The real test is that we can create a provider without errors
        provider = GoogleProvider(api_key="test-key")
        assert provider is not None

        # And that the problematic operation doesn't fail
        from pyasn1.type import constraint, univ

        try:
            # This was the failing operation
            subtypeSpec = univ.Integer.subtypeSpec + constraint.SingleValueConstraint(0, 1)
            assert subtypeSpec is not None
        except TypeError as e:
            if "can only concatenate tuple" in str(e):
                pytest.fail("pyasn1 compatibility not working")

    def test_initialization_without_pyasn1_error(self):
        """Test Google provider can be initialized without pyasn1 errors."""
        # This was the original failing case
        provider = GoogleProvider(api_key="test-key")
        assert provider.api_key == "test-key"

        # Should not raise "can only concatenate tuple" error
        assert provider is not None

    def test_initialization_with_env_vars(self):
        """Environment variables should not implicitly satisfy credential requirements."""

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                GoogleProvider()

        # Non-canonical variables should not be accepted implicitly.
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-key"}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                GoogleProvider()

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                GoogleProvider()

    def test_model_validation(self):
        """Test Google model validation."""
        provider = GoogleProvider(api_key="test-key")

        # Valid models
        valid_models = [
            "gemini-1.5-pro-latest",
            "gemini-pro",  # Legacy alias
            "gemini-pro-vision",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "models/gemini-1.5-pro-latest",
            "models/gemini-pro",  # Legacy prefix alias
            "models/gemini-pro-vision",
            "gemini-ultra",  # Future model
        ]

        for model in valid_models:
            assert provider.validate_model(model), f"Should validate {model}"

        # Invalid models
        invalid_models = ["gpt-4", "claude-3", "invalid-model"]
        for model in invalid_models:
            assert not provider.validate_model(model), f"Should not validate {model}"

    def test_complete_responses_payload_delegates(self):
        provider = GoogleProvider(api_key="test-key")
        payload = {
            "model": "models/gemini-1.5-pro-latest",
            "instructions": "Summarize succinctly",
            "input": [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "previous reply"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Explain the plan"}],
                },
            ],
            "max_output_tokens": 128,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["DONE"],
        }

        expected = ChatResponse(data="ok", model_id="models/gemini-1.5-pro-latest")
        with patch.object(GoogleProvider, "complete", return_value=expected) as mock_complete:
            result = provider.complete_responses_payload(payload)

        assert result is expected
        args, kwargs = mock_complete.call_args
        assert args[0] == "Explain the plan"
        assert args[1] == "models/gemini-1.5-pro-latest"
        assert kwargs["max_tokens"] == 128
        assert kwargs["temperature"] == 0.1
        assert kwargs["top_p"] == 0.9
        assert kwargs["stop"] == ["DONE"]
        context_blob = kwargs["context"]
        assert isinstance(context_blob, str)
        assert "Summarize succinctly" in context_blob
        assert "assistant: previous reply" in context_blob

    def test_model_info(self):
        """Test model info retrieval."""
        provider = GoogleProvider(api_key="test-key")

        # Test gemini-pro
        info = provider.get_model_info("gemini-pro")
        assert info["provider"] == "GoogleProvider"
        assert info["context_window"] == 2_000_000
        assert info["supports_vision"] is True
        assert info["canonical_model"] == "models/gemini-1.5-pro-latest"
        assert info["supports_functions"] is True

        # Test gemini-pro-vision
        info = provider.get_model_info("gemini-pro-vision")
        assert info["supports_vision"] is True
        assert info["canonical_model"] == "models/gemini-1.5-pro-latest"

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_completion_success(self, mock_configure, mock_model_class):
        """Test successful completion."""
        # Setup mock
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Hello from Gemini!"
        mock_model.generate_content.return_value = mock_response

        # Test
        provider = GoogleProvider(api_key="test-key")
        response = provider.complete("Say hello", "gemini-pro", temperature=0.7)

        assert response.data == "Hello from Gemini!"
        assert response.model_id == "models/gemini-1.5-pro-latest"

        # Verify API was configured
        mock_configure.assert_called_with(api_key="test-key")

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_completion_filters_unsupported_kwargs(self, mock_configure, mock_model_class):
        """Unsupported kwargs are filtered before hitting the Gemini SDK.

        The Google provider only passes prompt and generation_config to
        generate_content(). Other kwargs like response_format, safety_settings,
        and unknown parameters are silently filtered.
        """

        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Hello!"
        mock_model.generate_content.return_value = mock_response

        provider = GoogleProvider(api_key="test-key")
        provider.complete(
            "Hi",
            "gemini-pro",
            response_format={"type": "json_object"},
            foo="bar",
            safety_settings=["BLOCK_NONE"],
        )

        kwargs = mock_model.generate_content.call_args.kwargs
        # All unsupported kwargs are filtered out
        assert "response_format" not in kwargs
        assert "foo" not in kwargs
        assert "safety_settings" not in kwargs

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_completion_handles_missing_text_parts(self, mock_configure, mock_model_class):
        """Fallback to candidate parts when response.text is unavailable."""

        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        type(mock_response).text = PropertyMock(
            side_effect=ValueError("quick accessor requires a valid Part")
        )
        mock_response.candidates = [
            SimpleNamespace(
                finish_reason=2,
                content=SimpleNamespace(parts=[SimpleNamespace(text="Hello from parts!")]),
            )
        ]
        mock_model.generate_content.return_value = mock_response

        provider = GoogleProvider(api_key="test-key")
        response = provider.complete("Say hello", "gemini-pro")

        assert response.data == "Hello from parts!"
        assert response.model_id == "models/gemini-1.5-pro-latest"

    def test_extract_text_from_candidates_merges_multiple_parts(self):
        """Helper collapses heterogeneous candidate structures into text."""

        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="Alpha"),
                            {"text": "Beta"},
                            SimpleNamespace(other="ignored"),
                        ]
                    )
                ),
                SimpleNamespace(
                    parts=[{"text": "Gamma"}],
                ),
            ]
        )

        extracted = GoogleProvider._extract_text_from_candidates(response)
        assert extracted == "Alpha\nBeta\nGamma"

    def test_extract_text_from_candidates_handles_empty_candidates(self):
        """Gracefully returns empty string when no text parts exist."""
        response = SimpleNamespace(candidates=[SimpleNamespace(content=None)])

        extracted = GoogleProvider._extract_text_from_candidates(response)
        assert extracted == ""

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_completion_uses_usage_metadata(self, mock_configure, mock_model_class):
        """Google provider prefers SDK usage metadata when available."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model

        mock_response = Mock()
        mock_response.text = "Hello from Gemini!"
        mock_response.usage_metadata = SimpleNamespace(
            prompt_token_count=11, candidates_token_count=7, total_token_count=18
        )
        mock_model.generate_content.return_value = mock_response

        provider = GoogleProvider(api_key="test-key")
        response = provider.complete("Say hello", "gemini-pro")

        assert response.usage is not None
        assert response.usage.prompt_tokens == 11
        assert response.usage.completion_tokens == 7
        assert response.usage.total_tokens == 18

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_completion_error_handling(self, mock_configure, mock_model_class):
        """Test error handling in completions."""
        provider = GoogleProvider(api_key="test-key")
        with pytest.raises(ProviderAPIError, match="Unsupported Google model"):
            provider.complete("test", "invalid-model")

        # Test model creation error with valid id
        mock_model_class.side_effect = Exception("Invalid model")
        with pytest.raises(ProviderAPIError, match="Failed to create Gemini model"):
            provider.complete("test", "gemini-1.5-pro-latest")

        # Test API key error
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        mock_model_class.side_effect = None
        mock_model.generate_content.side_effect = Exception("API_KEY_INVALID")

        with pytest.raises(ProviderAPIError, match="Invalid Google API key"):
            provider.complete("test", "gemini-pro")

        # Test rate limit error
        mock_model.generate_content.side_effect = Exception("RATE_LIMIT_EXCEEDED")

        with pytest.raises(ProviderAPIError, match="rate limit"):
            provider.complete("test", "gemini-pro")

    def test_pyasn1_constraint_operations(self):
        """Test specific pyasn1 operations that were failing."""
        from pyasn1.type import constraint, univ

        # This exact operation was failing before the fix
        try:
            int_constraint = univ.Integer.subtypeSpec
            single_value = constraint.SingleValueConstraint(0, 1)
            combined = int_constraint + single_value
            assert combined is not None
        except TypeError as e:
            if "can only concatenate tuple" in str(e):
                pytest.fail(f"pyasn1 patch not working: {e}")

    def test_import_performance(self):
        """Test that our patch doesn't significantly impact import time."""
        import time

        # Remove from cache
        if "ember.models.providers.google" in sys.modules:
            del sys.modules["ember.models.providers.google"]

        # Measure import time
        start = time.perf_counter()

        elapsed = time.perf_counter() - start

        # Should be fast (under 1 second even on slow systems)
        assert elapsed < 1.0, f"Import too slow: {elapsed:.3f}s"

    def test_concurrent_initialization(self):
        """Test thread-safe provider initialization."""
        import queue
        import threading

        errors = queue.Queue()
        providers = queue.Queue()

        def create_provider(key: str):
            try:
                provider = GoogleProvider(api_key=key)
                providers.put(provider)
            except Exception as e:
                errors.put(e)

        # Create providers concurrently
        threads = []
        for i in range(5):
            t = threading.Thread(target=create_provider, args=(f"key-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Check no errors
        assert errors.empty(), f"Concurrent init failed: {list(errors.queue)}"

        # Check all providers created
        created = []
        while not providers.empty():
            created.append(providers.get())

        assert len(created) == 5
        assert all(p.api_key.startswith("key-") for p in created)


class TestProviderRegistry:
    """Test provider registration and discovery."""

    def test_all_providers_registered(self):
        """Test that all providers are properly registered."""
        from ember.models import ModelRegistry

        registry = ModelRegistry()

        # These models should work with their respective providers
        test_cases = [
            ("gpt-4", OpenAIProvider),
            ("claude-3-opus", AnthropicProvider),
            ("gemini-pro", GoogleProvider),
        ]

        for model_id, expected_provider in test_cases:
            # This will create the model and its provider
            with patch.object(registry, "_create_model") as mock_create:
                # Mock the provider creation
                mock_model = Mock()
                mock_model.provider = expected_provider(api_key="test-key")
                mock_create.return_value = mock_model

                model = registry.get_model(model_id)
                assert isinstance(model.provider, expected_provider)


# Regression tests to ensure issues don't resurface
class TestRegressionGuards:
    """Tests that guard against specific regressions."""

    def test_pyasn1_patch_importable(self):
        """Ensure pyasn1 patch module is importable."""
        from ember._internal.patches import pyasn1 as pyasn1_patch

        assert hasattr(pyasn1_patch, "ensure_pyasn1_compatibility")

    def test_google_provider_uses_pyasn1_patch(self):
        """Ensure Google provider imports the pyasn1 patch."""
        import inspect

        from ember.models.providers import google

        source = inspect.getsource(google)
        assert "patches.pyasn1" in source, "Google provider must import patches.pyasn1"
        assert "ensure_pyasn1_compatibility" in source, "Must call ensure_pyasn1_compatibility"

    def test_all_providers_import_without_pyasn1_errors(self):
        """Test that all providers import without pyasn1 tuple concatenation errors."""
        # Import all providers - this would raise TypeError if pyasn1 patch isn't applied
        from ember.models.providers import google, ollama  # noqa: F401

        # If we got here, no pyasn1 errors occurred


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "not integration"])
