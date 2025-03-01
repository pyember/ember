"""Unit tests for configuration functions in settings.py.

This module tests the deep_merge and resolve_env_vars functions.
"""

from typing import Any, Dict

import os
import pytest

from ember.core.registry.model.config.settings import (
    deep_merge,
    resolve_env_vars,
    EmberSettings,
    initialize_ember,
)


def test_deep_merge_dicts() -> None:
    """Test deep_merge with nested dictionaries."""
    base: Dict[str, Any] = {"a": 1, "b": {"x": 10, "y": 20}}
    override: Dict[str, Any] = {"b": {"y": 200, "z": 300}, "c": 3}
    expected: Dict[str, Any] = {"a": 1, "b": {"x": 10, "y": 200, "z": 300}, "c": 3}
    result = deep_merge(base=base, override=override)
    assert result == expected


def test_deep_merge_lists() -> None:
    """Test deep_merge with lists."""
    base = [1, 2, 3]
    override = [4, 5]
    expected = [1, 2, 3, 4, 5]
    result = deep_merge(base=base, override=override)
    assert result == expected


def test_deep_merge_mixed() -> None:
    """Test deep_merge with mixed types (dict and list)."""
    base = {"a": [1, 2], "b": "old"}
    override = {"a": [3], "b": "new"}
    expected = {"a": [1, 2, 3], "b": "new"}
    result = deep_merge(base=base, override=override)
    assert result == expected


def test_resolve_env_vars_with_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variable placeholders are correctly resolved."""
    monkeypatch.setenv("TEST_VAR", "resolved_value")
    data: Any = {"key": "${TEST_VAR}", "unchanged": "no_placeholder"}
    expected = {"key": "resolved_value", "unchanged": "no_placeholder"}
    result = resolve_env_vars(data=data)
    assert result == expected


def test_resolve_env_vars_nested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that resolve_env_vars works for nested data structures."""
    monkeypatch.setenv("NESTED_VAR", "nested")
    data = {"outer": {"inner": "${NESTED_VAR}"}, "list": ["${NESTED_VAR}", "static"]}
    expected = {"outer": {"inner": "nested"}, "list": ["nested", "static"]}
    result = resolve_env_vars(data=data)
    assert result == expected


def test_ember_settings_load_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable loading."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key_123")
    settings = EmberSettings()
    assert settings.openai_api_key == "test_key_123"


# def test_ember_settings_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
#     """Tests that EmberSettings correctly loads from env vars and YAML files."""
#     config_content = """
#     registry:
#       auto_register: true
#       auto_discover: false
#     other:
#       debug: true
#     """
#     config_path = tmp_path / "config.yaml"
#     config_path.write_text(config_content)

#     monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
#     monkeypatch.setenv("ANTHROPIC_API_KEY", "test_anthropic_key")

#     settings = EmberSettings(model_config_path=str(config_path))
#     assert settings.openai_api_key == "test_openai_key"
#     assert settings.anthropic_api_key == "test_anthropic_key"
#     assert settings.registry.auto_register is True
#     assert settings.registry.auto_discover is False
#     assert settings.other.debug is True


# def test_initialize_ember(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
#     """Tests initialize_ember for full configuration loading and merging."""
#     main_config = """
#     registry:
#       included_configs:
#         - "{}/openai_config.yaml"
#       models:
#         - model_id: "local:model"
#           model_name: "Local Model"
#           cost:
#             input_cost_per_thousand: 1.0
#             output_cost_per_thousand: 2.0
#           rate_limit:
#             tokens_per_minute: 1000
#             requests_per_minute: 100
#           provider:
#             name: "LocalProvider"
#             default_api_key: "local_key"
#           api_key: "local_key"
#     """
#     main_config_path = tmp_path / "config.yaml"
#     main_config_path.write_text(main_config.format(tmp_path))

#     openai_config = """
#     models:
#       - model_id: "openai:gpt-4o"
#         model_name: "GPT-4o"
#         cost:
#           input_cost_per_thousand: 5.0
#           output_cost_per_thousand: 15.0
#         rate_limit:
#           tokens_per_minute: 10000
#           requests_per_minute: 1000
#         provider:
#           name: "OpenAI"
#           default_api_key: "${OPENAI_API_KEY}"
#         api_key: null
#     """
#     openai_config_path = tmp_path / "openai_config.yaml"
#     openai_config_path.write_text(openai_config)

#     monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")

#     registry = initialize_ember(
#         config_path=str(main_config_path), auto_register=True, auto_discover=False
#     )
#     model_ids = registry.list_models()
#     assert "local:model" in model_ids
#     assert "openai:gpt-4o" in model_ids
#     openai_model_info = registry.get_model_info("openai:gpt-4o")
#     assert openai_model_info.get_api_key() == "test_openai_key"


# def test_initialize_ember_missing_config(tmp_path: Path) -> None:
#     """Tests error handling for missing configuration file."""
#     with pytest.raises(EmberError):
#         initialize_ember(config_path=str(tmp_path / "nonexistent.yaml"))


# def test_initialize_ember_invalid_yaml(tmp_path: Path) -> None:
#     """Tests error handling for invalid YAML content."""
#     invalid_config = "registry: [invalid_yaml"
#     config_path = tmp_path / "config.yaml"
#     config_path.write_text(invalid_config)
#     with pytest.raises(EmberError):
#         initialize_ember(config_path=str(config_path))
