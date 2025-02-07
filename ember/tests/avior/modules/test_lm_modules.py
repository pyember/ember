import pytest
from ember.registry.operators.operator_base import LMModule, LMModuleConfig

# from ... import get_model_registry if needed


@pytest.mark.parametrize("model_name", ["gpt-4o", "gpt-4-turbo"])
def test_lm_module_valid_model(model_name, mock_lm_generation):
    """
    Test LMModule with valid models: verifies initialization and responses.
    """
    config = LMModuleConfig(model_name=model_name, temperature=1.0)
    # TODO: Create a model_registry instance as needed if not global.
    from ember.registry.model.model_registry import ModelRegistry

    lm = LMModule(config, model_registry=ModelRegistry())
    response = lm("Hello")
    assert "Mocked response: Hello" in response
    assert "temp=1.0" in response


def test_lm_module_invalid_model():
    """
    Invalid model name should raise ValueError.
    """
    config = LMModuleConfig(model_name="not_registered_model")
    from ember.registry.model.model_registry import ModelRegistry

    with pytest.raises(ValueError):
        LMModule(config, model_registry=ModelRegistry())


@pytest.mark.parametrize("temp", [-1.0, 2.1])
def test_lm_module_invalid_temperature(temp):
    """
    Invalid temperature out of [0,2] range should cause ValidationError on LMModuleConfig.
    """
    with pytest.raises(ValueError):
        LMModuleConfig(model_name="gpt-4o", temperature=temp)


def test_lm_module_persona_max_tokens(mock_lm_generation):
    """
    Test LMModule with persona and max_tokens set.
    """
    config = LMModuleConfig(
        model_name="gpt-4o", temperature=0.5, max_tokens=100, persona="friendly"
    )
    from ember.registry.model.model_registry import ModelRegistry

    lm = LMModule(config, model_registry=ModelRegistry())
    response = lm("Hello")
    assert "Mocked response: Hello" in response
