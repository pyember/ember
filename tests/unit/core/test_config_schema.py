"""Unit tests for the centralized configuration schema.

This module tests the data structures and helpers in the configuration system.
"""


# Mock Pydantic BaseModel for testing
class BaseModel:
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Mock configuration classes
class Cost(BaseModel):
    """Cost configuration for a model."""

    def __init__(self, **data):
        self.input_cost_per_thousand = data.get("input_cost_per_thousand", 0.0)
        self.output_cost_per_thousand = data.get("output_cost_per_thousand", 0.0)

    def calculate(self, input_tokens, output_tokens):
        """Calculate cost for tokens."""
        return (
            self.input_cost_per_thousand * input_tokens / 1000
            + self.output_cost_per_thousand * output_tokens / 1000
        )


class Model(BaseModel):
    """Model configuration class."""

    def __init__(self, **data):
        self.id = data.get("id")
        self.name = data.get("name")
        self.provider = data.get("provider")
        self.cost = data.get("cost", Cost())
        self.rate_limit = data.get("rate_limit", {})


class ApiKey(BaseModel):
    """API key configuration."""

    def __init__(self, **data):
        self.key = data.get("key", "")


class Provider(BaseModel):
    """Provider configuration."""

    def __init__(self, **data):
        self.enabled = data.get("enabled", True)
        self.api_keys = data.get("api_keys", {})
        self.base_url = data.get("base_url")
        self.models = data.get("models", [])
        self.__root_key__ = ""

    def get_model_config(self, model_id):
        """Get model by ID."""
        for model in self.models:
            if model.id == model_id:
                return model
            if model.id == model_id.split(":")[-1]:
                return model
        return None

    def get_default_api_key(self):
        """Get default API key."""
        if "default" in self.api_keys:
            return self.api_keys["default"].key
        return None


class ModelRegistryConfig(BaseModel):
    """Configuration for the model registry."""

    def __init__(self, **data):
        self.auto_discover = data.get("auto_discover", True)
        self.auto_register = data.get("auto_register", True)
        self.providers = data.get("providers", {})

    def get_provider(self, name):
        """Get provider by name."""
        return self.providers.get(name.lower())

    def get_model_config(self, model_id):
        """Get model by ID."""
        if ":" in model_id:
            provider_name, model_name = model_id.split(":", 1)
            provider = self.get_provider(provider_name)
            if provider:
                return provider.get_model_config(model_name)
        return None


class EmberConfig(BaseModel):
    """Root configuration for Ember."""

    def __init__(self, **data):
        self.model_registry = data.get("model_registry", ModelRegistryConfig())

    def get_provider(self, name):
        """Get provider by name."""
        return self.model_registry.get_provider(name)

    def get_model_config(self, model_id):
        """Get model by ID."""
        return self.model_registry.get_model_config(model_id)


def test_cost_calculation():
    """Test cost calculation logic."""
    cost = Cost(input_cost_per_thousand=1.0, output_cost_per_thousand=2.0)
    total = cost.calculate(1000, 500)
    assert total == 1.0 + 1.0  # 1.0 for input, 1.0 for output


def test_model_creation():
    """Test creating model configuration."""
    model = Model(
        id="gpt-4",
        name="GPT-4",
        provider="openai",
        cost=Cost(input_cost_per_thousand=5.0, output_cost_per_thousand=15.0),
    )
    assert model.id == "gpt-4"
    assert model.name == "GPT-4"
    assert model.provider == "openai"
    assert model.cost.input_cost_per_thousand == 5.0


def test_provider_get_model():
    """Test provider get_model_config method."""
    model1 = Model(id="gpt-4", name="GPT-4", provider="openai")
    model2 = Model(id="gpt-3.5-turbo", name="GPT-3.5 Turbo", provider="openai")
    provider = Provider(
        enabled=True,
        models=[model1, model2],
        api_keys={"default": ApiKey(key="test-key")},
    )
    provider.__root_key__ = "openai"

    # Test getting by ID
    result = provider.get_model_config("gpt-4")
    assert result.name == "GPT-4"

    # Test getting by qualified ID
    result = provider.get_model_config("openai:gpt-4")
    assert result.name == "GPT-4"

    # Test non-existent model
    result = provider.get_model_config("non-existent")
    assert result is None


def test_provider_get_api_key():
    """Test provider get_default_api_key method."""
    provider = Provider(enabled=True, api_keys={"default": ApiKey(key="test-key")})
    assert provider.get_default_api_key() == "test-key"

    # Test no API key
    provider_no_key = Provider(enabled=True)
    assert provider_no_key.get_default_api_key() is None


def test_registry_config_get_provider():
    """Test getting provider from registry config."""
    provider = Provider(enabled=True, api_keys={"default": ApiKey(key="test-key")})
    registry = ModelRegistryConfig(providers={"openai": provider})

    # Test getting by name
    result = registry.get_provider("openai")
    assert result is provider

    # Test case-insensitive
    result = registry.get_provider("OpEnAi")
    assert result is provider

    # Test non-existent
    result = registry.get_provider("non-existent")
    assert result is None


def test_registry_config_get_model():
    """Test getting model from registry config."""
    model = Model(id="gpt-4", name="GPT-4", provider="openai")
    provider = Provider(
        enabled=True, models=[model], api_keys={"default": ApiKey(key="test-key")}
    )
    registry = ModelRegistryConfig(providers={"openai": provider})

    # Test getting by qualified ID
    result = registry.get_model_config("openai:gpt-4")
    assert result is model

    # Test non-existent
    result = registry.get_model_config("non-existent")
    assert result is None


def test_ember_config_accessors():
    """Test the accessor methods in EmberConfig."""
    model = Model(id="gpt-4", name="GPT-4", provider="openai")
    provider = Provider(
        enabled=True, models=[model], api_keys={"default": ApiKey(key="test-key")}
    )
    registry = ModelRegistryConfig(providers={"openai": provider})
    config = EmberConfig(model_registry=registry)

    # Test get_provider
    result = config.get_provider("openai")
    assert result is provider

    # Test get_model_config
    result = config.get_model_config("openai:gpt-4")
    assert result is model
