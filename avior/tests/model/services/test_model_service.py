import pytest
from unittest.mock import MagicMock

from src.avior.registry.model.services.model_service import ModelService
from src.avior.registry.model.services.usage_service import UsageService
from src.avior.registry.model.registry.model_registry import ModelRegistry
from src.avior.registry.model.provider_registry.base import BaseProviderModel
from src.avior.registry.model.schemas.chat_schemas import ChatResponse
from src.avior.registry.model.schemas.usage import UsageStats
from src.avior.registry.model.schemas.model_info import ModelInfo
from src.avior.registry.model.schemas.provider_info import ProviderInfo
from src.avior.registry.model.schemas.cost import ModelCost, RateLimit


class MockProvider(BaseProviderModel):
    def __init__(self):
        # Attach a valid ModelInfo to avoid None-type issues.
        model_info = ModelInfo(
            model_id="mock:mock-model",
            model_name="mock-model",
            cost=ModelCost(),
            rate_limit=RateLimit(),
            provider=ProviderInfo(name="MockProvider", default_api_key="fake_key"),
            api_key="fake_key",
        )
        super().__init__(model_info)

    def create_client(self):
        return "mock_client"

    def forward(self, request):
        raw_output = {
            "usage": {"total_tokens": 10, "prompt_tokens": 3, "completion_tokens": 7}
        }
        usage_stats = UsageStats(**raw_output["usage"])
        return ChatResponse(
            data=f"Mock output for: {request.prompt}",
            raw_output=raw_output,
            usage=usage_stats,
        )

    def calculate_usage(self, raw_output):
        return raw_output["usage"]


@pytest.fixture
def setup_services():
    reg = ModelRegistry()
    usage_svc = UsageService()
    svc = ModelService(reg, usage_svc)
    return reg, usage_svc, svc


def test_model_service_forward_query(setup_services):
    reg, usage_svc, svc = setup_services
    # Now instantiate the mock with a valid ModelInfo
    reg._models["mock-model"] = MockProvider()
    resp = svc.invoke_model("mock-model", "Hello test!")
    assert resp.data == "Mock output for: Hello test!"

    summary = usage_svc.get_usage_summary("mock:mock-model")
    assert summary.total_tokens_used == 10


def test_model_service_call(setup_services):
    reg, usage_svc, svc = setup_services
    reg._models["mock-model"] = MockProvider()
    resp = svc("mock-model", "Another test")
    assert resp.data == "Mock output for: Another test"


def test_model_service_not_found(setup_services):
    """
    If parse_model_str fails, it sets validated_id to the same value but eventually
    raises "Model 'X' not found." We can change our test to match that message rather
    than "is not a valid model string."
    """
    _, _, svc = setup_services
    with pytest.raises(ValueError, match="not found"):
        svc("not-registered", "No model here")
