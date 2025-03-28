"""Tests for the configuration schema module.

This module contains tests for the configuration schema classes in ember.core.config.schema.
"""


from ember.core.config.schema import Cost, EmberConfig, Model, Provider, RegistryConfig


class TestCost:
    """Tests for the Cost value object."""

    def test_creation(self):
        """Test creating a Cost object with default values."""
        cost = Cost()
        assert cost.input_cost_per_thousand == 0.0
        assert cost.output_cost_per_thousand == 0.0

    def test_creation_with_values(self):
        """Test creating a Cost object with custom values."""
        cost = Cost(input_cost_per_thousand=1.5, output_cost_per_thousand=2.5)
        assert cost.input_cost_per_thousand == 1.5
        assert cost.output_cost_per_thousand == 2.5

    def test_calculate(self):
        """Test cost calculation."""
        cost = Cost(input_cost_per_thousand=2.0, output_cost_per_thousand=3.0)
        calculated = cost.calculate(input_tokens=100, output_tokens=50)
        # (2.0 * 100 + 3.0 * 50) / 1000 = 0.35
        assert calculated == 0.35


class TestModel:
    """Tests for the Model configuration class."""

    def test_minimal_creation(self):
        """Test creating a Model with minimal required fields."""
        model = Model(id="test", name="Test Model", provider="test-provider")
        assert model.id == "test"
        assert model.name == "Test Model"
        assert model.provider == "test-provider"
        assert model.cost_input == 0.0
        assert model.cost_output == 0.0
        assert model.cost.input_cost_per_thousand == 0.0
        assert model.cost.output_cost_per_thousand == 0.0

    def test_full_creation(self):
        """Test creating a Model with all fields."""
        model = Model(
            id="test",
            name="Test Model",
            provider="test-provider",
            cost_input=1.5,
            cost_output=2.5,
            rate_limit={"tokens_per_minute": 1000, "requests_per_minute": 100},
        )
        assert model.id == "test"
        assert model.name == "Test Model"
        assert model.provider == "test-provider"
        assert model.cost.input_cost_per_thousand == 1.5
        assert model.cost.output_cost_per_thousand == 2.5
        assert model.rate_limit.tokens_per_minute == 1000

    def test_cost_property(self):
        """Test the cost property."""
        model = Model(
            id="test",
            name="Test Model",
            provider="test-provider",
            cost_input=1.5,
            cost_output=2.5,
        )
        assert isinstance(model.cost, Cost)
        assert model.cost.input_cost_per_thousand == 1.5
        assert model.cost.output_cost_per_thousand == 2.5

    def test_arbitrary_fields(self):
        """Test adding arbitrary fields."""
        model = Model(
            id="test",
            name="Test Model",
            provider="test-provider",
            context_length=4096,
            vision_enabled=True,
        )
        assert model.id == "test"
        assert model.name == "Test Model"
        assert model.provider == "test-provider"
        assert model.context_length == 4096
        assert model.vision_enabled is True


class TestProvider:
    """Tests for the Provider configuration class."""

    def test_minimal_creation(self):
        """Test creating a Provider with minimal fields."""
        provider = Provider()
        assert provider.enabled is True
        assert provider.api_keys == {}
        assert provider.models == {}

    def test_full_creation(self):
        """Test creating a Provider with all fields."""
        models = [
            Model(id="model1", name="Model One", provider="test"),
            Model(id="model2", name="Model Two", provider="test"),
        ]
        provider = Provider(
            enabled=True, api_keys={"default": {"key": "test-api-key"}}, models=models
        )
        assert provider.enabled is True
        assert provider.api_keys["default"]["key"] == "test-api-key"
        assert len(provider.models) == 2
        assert provider.models["model1"].name == "Model One"

    def test_get_model_config(self):
        """Test get_model_config method."""
        models = [
            Model(id="model1", name="Model One", provider="test"),
            Model(id="model2", name="Model Two", provider="test"),
        ]
        provider = Provider(
            enabled=True, api_keys={"default": {"key": "test-api-key"}}, models=models
        )
        provider.__root_key__ = "test"

        # Get by short name
        model = provider.get_model_config("model1")
        assert model is not None
        assert model.name == "Model One"

        # Get by full name
        model = provider.get_model_config("test:model1")
        assert model is not None
        assert model.name == "Model One"

        # Non-existent model
        model = provider.get_model_config("non-existent")
        assert model is None

    def test_arbitrary_fields(self):
        """Test adding arbitrary fields."""
        provider = Provider(
            enabled=True,
            api_keys={"default": {"key": "test-api-key"}},
            base_url="https://api.example.com",
            timeout=30.0,
        )
        assert provider.enabled is True
        assert provider.api_keys["default"]["key"] == "test-api-key"
        assert provider.base_url == "https://api.example.com"
        assert provider.timeout == 30.0


class TestRegistryConfig:
    """Tests for the RegistryConfig class."""

    def test_minimal_creation(self):
        """Test creating a RegistryConfig with minimal fields."""
        config = RegistryConfig()
        assert config.auto_discover is True
        assert config.providers == {}

    def test_full_creation(self):
        """Test creating a RegistryConfig with all fields."""
        providers = {
            "provider1": Provider(enabled=True, api_keys={"default": {"key": "key1"}}),
            "provider2": Provider(enabled=False, api_keys={"default": {"key": "key2"}}),
        }
        config = RegistryConfig(auto_discover=False, providers=providers)
        assert config.auto_discover is False
        assert len(config.providers) == 2
        assert config.providers["provider1"].enabled is True
        assert config.providers["provider2"].enabled is False

    def test_provider_root_key(self):
        """Test that provider root keys are set correctly."""
        providers = {
            "provider1": Provider(enabled=True),
            "provider2": Provider(enabled=False),
        }
        config = RegistryConfig(providers=providers)
        assert config.providers["provider1"].__root_key__ == "provider1"
        assert config.providers["provider2"].__root_key__ == "provider2"

    def test_arbitrary_fields(self):
        """Test adding arbitrary fields."""
        config = RegistryConfig(
            auto_discover=True, cache_ttl=3600, fallback_provider="openai"
        )
        assert config.auto_discover is True
        assert config.cache_ttl == 3600
        assert config.fallback_provider == "openai"


class TestEmberConfig:
    """Tests for the EmberConfig class."""

    def test_minimal_creation(self):
        """Test creating an EmberConfig with minimal fields."""
        config = EmberConfig()
        assert isinstance(config.registry, RegistryConfig)
        assert config.registry.auto_discover is True

    def test_full_creation(self):
        """Test creating an EmberConfig with all fields."""
        registry = RegistryConfig(
            auto_discover=False,
            providers={
                "provider1": Provider(enabled=True, api_key="key1"),
                "provider2": Provider(enabled=False, api_key="key2"),
            },
        )
        config = EmberConfig(registry=registry, logging={"level": "DEBUG"})
        assert config.registry.auto_discover is False
        assert len(config.registry.providers) == 2
        assert config.logging.level == "DEBUG"

    def test_get_provider(self):
        """Test get_provider method."""
        providers = {
            "provider1": Provider(enabled=True, api_key="key1"),
            "provider2": Provider(enabled=False, api_key="key2"),
        }
        registry = RegistryConfig(providers=providers)
        config = EmberConfig(registry=registry)

        # Get existing provider (case-insensitive)
        provider = config.get_provider("PROVIDER1")
        assert provider is not None
        assert provider.api_key == "key1"

        # Non-existent provider
        provider = config.get_provider("non-existent")
        assert provider is None

    def test_get_model_config(self):
        """Test get_model_config method."""
        models1 = [
            Model(id="model1", name="Model One", provider="provider1"),
            Model(id="model2", name="Model Two", provider="provider1"),
        ]
        models2 = [Model(id="model3", name="Model Three", provider="provider2")]
        providers = {
            "provider1": Provider(
                enabled=True, api_keys={"default": {"key": "key1"}}, models=models1
            ),
            "provider2": Provider(
                enabled=True, api_keys={"default": {"key": "key2"}}, models=models2
            ),
        }
        registry = RegistryConfig(providers=providers)
        config = EmberConfig(registry=registry)

        # Get existing model
        model = config.get_model_config("provider1:model1")
        assert model is not None
        assert model.name == "Model One"

        # Get model from second provider
        model = config.get_model_config("provider2:model3")
        assert model is not None
        assert model.name == "Model Three"

        # Non-existent model
        model = config.get_model_config("provider1:non-existent")
        assert model is None

        # Invalid format
        model = config.get_model_config("invalid-format")
        assert model is None

    def test_arbitrary_fields(self):
        """Test adding arbitrary fields."""
        config = EmberConfig(
            custom_section={"enabled": True, "value": 42},
            experimental_features=["feature1", "feature2"],
        )
        assert config.custom_section["enabled"] is True
        assert config.custom_section["value"] == 42
        assert len(config.experimental_features) == 2
