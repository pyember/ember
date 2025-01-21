import pytest
from abc import ABC, abstractmethod
from src.avior.registry.model.provider_registry.base import BaseProviderModel
from src.avior.registry.model.schemas.usage import UsageStats
from src.avior.registry.model.schemas.chat_schemas import ChatResponse


def test_model_response_creation():
    response = ChatResponse(
        data="Hello World",
        raw_output={"foo": "bar"},
        usage=UsageStats(total_tokens=42),
    )
    assert response.data == "Hello World"
    assert response.raw_output == {"foo": "bar"}
    assert response.usage.model_dump(exclude_none=True) == {
        "total_tokens": 42,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }


def test_base_provider_instantiation():
    # BaseProviderModel is abstract; should not be instantiated directly.
    with pytest.raises(TypeError):
        BaseProviderModel(model_info=None)


def test_base_provider_model_call():
    """
    Create a mock subclass to test that __call__ delegates to forward().
    """

    class MockProvider(BaseProviderModel):
        def create_client(self):
            return "mock_client"

        def forward(self, request):
            return ChatResponse(
                data=f"Mocked result for: {request.prompt}",
                raw_output={"info": "some raw output"},
                usage=UsageStats(total_tokens=0),
            )

        def calculate_usage(self, raw_output):
            return {"total_tokens": 0}

    provider = MockProvider(model_info=None)
    response = provider("Test prompt")
    assert response.data == "Mocked result for: Test prompt"
    assert response.raw_output == {"info": "some raw output"}
    assert response.usage.model_dump(exclude_none=True) == {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
    }
