"""Minimal integration test for configuration system.

This module verifies the configuration loading, validation, and integration with
the model registry system. It tests both the basic YAML loading functionality and
the centralized configuration system.
"""

import os
import tempfile
from contextlib import contextmanager
from unittest.mock import patch
import yaml
import pytest

# Import modules for centralized configuration testing
try:
    from ember.core.configs.schema import EmberConfig, ModelConfig, ProviderConfig, ApiKeyConfig, CostConfig
    from ember.core.configs.config_manager import ConfigManager
    from ember.core.registry.model.initialization import initialize_registry
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


# Test YAML configuration
TEST_CONFIG = """
model_registry:
  auto_discover: false
  auto_register: true
  providers:
    openai:
      enabled: true
      api_keys:
        default:
          key: "test-openai-key"
      models:
        - id: "gpt-4"
          name: "GPT-4"
          cost:
            input_cost_per_thousand: 5.0
            output_cost_per_thousand: 15.0
          rate_limit:
            tokens_per_minute: 100000
            requests_per_minute: 500
"""

class MockModelInfo:
    """Simple mock class to test configuration loading."""
    def __init__(self, model_id, name, cost, provider, api_key):
        self.model_id = model_id
        self.name = name
        self.cost = cost
        self.provider = provider
        self.api_key = api_key
    
    def get_api_key(self):
        return self.api_key


def test_basic_yaml_loading():
    """Test basic YAML loading functionality."""
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(TEST_CONFIG)
        config_path = f.name
    
    try:
        # Load the YAML file
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        
        # Basic assertions
        assert "model_registry" in config_data
        assert config_data["model_registry"]["auto_discover"] is False
        assert "providers" in config_data["model_registry"]
        assert "openai" in config_data["model_registry"]["providers"]
        
        # Check provider details
        provider = config_data["model_registry"]["providers"]["openai"]
        assert provider["enabled"] is True
        assert provider["api_keys"]["default"]["key"] == "test-openai-key"
        
        # Check model details
        models = provider["models"]
        assert len(models) == 1
        assert models[0]["id"] == "gpt-4"
        assert models[0]["name"] == "GPT-4"
        assert models[0]["cost"]["input_cost_per_thousand"] == 5.0
        
    finally:
        # Clean up the temporary file
        os.unlink(config_path)


def test_environment_variable_substitution():
    """Test environment variable substitution in config."""
    # Create modified config with environment variable reference
    config_with_env = TEST_CONFIG.replace(
        'key: "test-openai-key"', 
        'key: "${TEST_API_KEY}"'
    )
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_with_env)
        config_path = f.name
    
    try:
        # Set environment variable
        os.environ["TEST_API_KEY"] = "key-from-environment"
        
        # Basic environment variable substitution
        def substitute_env(data_str):
            """Substitute environment variables in string."""
            if not isinstance(data_str, str):
                return data_str
            
            if "${" in data_str:
                # Extract variable name
                start = data_str.find("${") + 2
                end = data_str.find("}", start)
                if start > 1 and end > start:
                    var_name = data_str[start:end]
                    # Get value from environment
                    var_value = os.environ.get(var_name, "")
                    # Replace in string
                    return data_str.replace(f"${{{var_name}}}", var_value)
            return data_str
        
        # Load and process the YAML file
        with open(config_path, "r") as f:
            config_raw = f.read()
            # Simple substitution - in real code would be more robust
            config_with_subs = substitute_env(config_raw)
            config_data = yaml.safe_load(config_with_subs)
        
        # Check that environment variable was substituted
        api_key = config_data["model_registry"]["providers"]["openai"]["api_keys"]["default"]["key"]
        assert api_key == "key-from-environment", f"Got unexpected API key: {api_key}"
        
    finally:
        # Clean up the temporary file
        os.unlink(config_path)
        # Clean up environment
        if "TEST_API_KEY" in os.environ:
            del os.environ["TEST_API_KEY"]


def test_model_info_creation():
    """Test creating model info objects from config."""
    # Load config data
    config_data = yaml.safe_load(TEST_CONFIG)
    
    # Extract model data
    model_data = config_data["model_registry"]["providers"]["openai"]["models"][0]
    provider_data = config_data["model_registry"]["providers"]["openai"]
    
    # Create a mock cost object
    class MockCost:
        def __init__(self, input_cost_per_thousand, output_cost_per_thousand):
            self.input_cost_per_thousand = input_cost_per_thousand
            self.output_cost_per_thousand = output_cost_per_thousand
    
    # Create a mock provider object
    class MockProvider:
        def __init__(self, name, default_api_key):
            self.name = name
            self.default_api_key = default_api_key
    
    # Create model info from config
    cost = MockCost(
        input_cost_per_thousand=model_data["cost"]["input_cost_per_thousand"],
        output_cost_per_thousand=model_data["cost"]["output_cost_per_thousand"]
    )
    
    provider = MockProvider(
        name="OpenAI",
        default_api_key=provider_data["api_keys"]["default"]["key"]
    )
    
    model_info = MockModelInfo(
        model_id=f"openai:{model_data['id']}",
        name=model_data["name"],
        cost=cost,
        provider=provider,
        api_key=provider_data["api_keys"]["default"]["key"]
    )
    
    # Verify model info
    assert model_info.model_id == "openai:gpt-4"
    assert model_info.name == "GPT-4"
    assert model_info.cost.input_cost_per_thousand == 5.0
    assert model_info.cost.output_cost_per_thousand == 15.0
    assert model_info.provider.name == "OpenAI"
    assert model_info.get_api_key() == "test-openai-key"


@contextmanager
def temp_config_file(content):
    """Create a temporary config file with the given content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        config_path = f.name
    
    try:
        yield config_path
    finally:
        # Clean up the temporary file
        os.unlink(config_path)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
def test_centralized_config_schema():
    """Test the centralized configuration schema objects."""
    # Create basic configuration objects
    api_key = ApiKeyConfig(key="test-key", org_id="test-org")
    assert api_key.key == "test-key"
    assert api_key.org_id == "test-org"
    
    cost = CostConfig(
        input_cost_per_thousand=5.0,
        output_cost_per_thousand=15.0
    )
    assert cost.input_cost_per_thousand == 5.0
    assert cost.output_cost_per_thousand == 15.0
    assert cost.input_cost_per_million == 5000.0  # Derived value
    
    model = ModelConfig(
        id="gpt-4",
        name="GPT-4",
        provider="openai",
        cost=cost
    )
    assert model.id == "gpt-4"
    assert model.name == "GPT-4"
    assert model.provider == "openai"
    assert model.cost.input_cost_per_thousand == 5.0
    
    provider = ProviderConfig(
        enabled=True,
        api_keys={"default": api_key},
        models=[model]
    )
    assert provider.enabled is True
    assert provider.api_keys["default"].key == "test-key"
    assert len(provider.models) == 1
    assert provider.models[0].id == "gpt-4"
    
    # Create full EmberConfig
    config = EmberConfig(
        model_registry={
            "providers": {
                "openai": provider
            }
        }
    )
    assert config.model_registry.providers["openai"].enabled is True
    assert config.model_registry.providers["openai"].models[0].name == "GPT-4"
    
    # Test helper methods
    retrieved_provider = config.get_provider("openai")
    assert retrieved_provider is not None
    assert retrieved_provider.enabled is True
    
    retrieved_model = config.get_model_config("openai:gpt-4")
    assert retrieved_model is not None
    assert retrieved_model.name == "GPT-4"


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
def test_config_to_registry_conversion():
    """Test converting configuration to model registry entries."""
    # Skip this test in environments that can't import necessary modules
    if not IMPORTS_AVAILABLE:
        pytest.skip("Required imports not available")
    
    # Create a test configuration
    config_content = """
    model_registry:
      auto_discover: false
      auto_register: true
      providers:
        openai:
          enabled: true
          api_keys:
            default:
              key: "test-openai-key"
          models:
            - id: "gpt-4"
              name: "GPT-4"
              cost:
                input_cost_per_thousand: 5.0
                output_cost_per_thousand: 15.0
    """
    
    with temp_config_file(config_content) as config_path:
        # Create a config manager with the test file
        config_manager = ConfigManager(config_path=config_path)
        config_manager.load()
        
        # Mock the model registry to avoid real API calls
        with patch('ember.core.registry.model.base.registry.model_registry.ModelRegistry') as MockRegistry:
            # Set up the mock
            mock_registry = MockRegistry.return_value
            mock_registry.is_registered.return_value = False
            mock_registry.register_model.return_value = None
            
            # Initialize registry from config
            registry = initialize_registry(
                config_manager=config_manager,
                auto_discover=False
            )
            
            # Verify the registry was initialized properly
            MockRegistry.assert_called_once()
            # Should register one model
            assert mock_registry.register_model.call_count == 1
            # Should not try to discover models
            assert mock_registry.discover_models.call_count == 0
            
            # Verify the model_info passed to register_model
            call_args = mock_registry.register_model.call_args[0]
            model_info = call_args[0]
            assert model_info.model_id == "openai:gpt-4"
            assert model_info.provider.name == "Openai"
            assert model_info.api_key == "test-openai-key"