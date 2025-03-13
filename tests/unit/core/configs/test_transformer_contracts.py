"""Tests for the ConfigTransformer interface contracts.

This module focuses on testing that the ConfigTransformer interface and
its implementations adhere to SOLID principles, particularly the
Composition, Open/Closed, and Single Responsibility principles.
"""

import os
from typing import Dict, Any, Callable, Protocol, runtime_checkable
from unittest import mock

import pytest

from src.ember.core.configs.transformer import (
    ConfigTransformer, 
    resolve_env_vars, 
    deep_merge, 
    add_default_values
)


@runtime_checkable
class TransformationFunction(Protocol):
    """A protocol that transformation functions must satisfy."""
    
    def __call__(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Transform the configuration.
        
        Args:
            config: The configuration to transform
            
        Returns:
            The transformed configuration
        """
        ...


class TestTransformerContracts:
    """Tests for the ConfigTransformer interface contracts."""

    def test_all_transformations_satisfy_protocol(self):
        """Test that all transformation functions satisfy the protocol."""
        # Check built-in transformation functions
        transformations = [
            resolve_env_vars,
            add_default_values,
            lambda config: deep_merge({}, config)  # Wrap deep_merge to match the protocol
        ]
        
        # Verify each transformation satisfies the protocol
        for transform in transformations:
            assert isinstance(transform, TransformationFunction), f"{transform.__name__} does not satisfy TransformationFunction protocol"
    
    def test_transformer_composition(self):
        """Test that transformers can be composed (Open/Closed Principle)."""
        # Create a transformer with multiple transformations
        transformer = ConfigTransformer()
        
        # Add custom transformations
        transformer.add_transformation(lambda config: {**config, "step1": True})
        transformer.add_transformation(lambda config: {**config, "step2": True})
        
        # Apply transformations
        result = transformer.transform({"original": True})
        
        # All transformations should be applied in order
        assert result == {"original": True, "step1": True, "step2": True}
        
        # Add another transformation without modifying existing code
        transformer.add_transformation(lambda config: {**config, "step3": True})
        
        # Apply again
        result = transformer.transform({"original": True})
        
        # New transformation should also be applied
        assert result == {"original": True, "step1": True, "step2": True, "step3": True}
    
    def test_transformer_single_responsibility(self):
        """Test that each transformation has a single responsibility."""
        # Each transformation should have a focused purpose
        
        # Test resolve_env_vars - should only handle environment variables
        with mock.patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            config = {"key": "${TEST_VAR}", "normal": "value"}
            result = resolve_env_vars(config)
            
            # Should resolve environment variables
            assert result["key"] == "test_value"
            # Should not affect other values
            assert result["normal"] == "value"
        
        # Test deep_merge - should only handle merging
        dict1 = {"a": 1, "b": {"c": 2}}
        dict2 = {"b": {"d": 3}, "e": 4}
        result = deep_merge(dict1, dict2)
        
        # Should merge dictionaries correctly
        assert result == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}
        # Should not modify the original dictionaries
        assert dict1 == {"a": 1, "b": {"c": 2}}
        assert dict2 == {"b": {"d": 3}, "e": 4}
    
    def test_transformers_are_pure_functions(self):
        """Test that transformations don't have side effects."""
        # Transformations should not modify their inputs
        
        # Test with resolve_env_vars
        with mock.patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            original = {"key": "${TEST_VAR}", "nested": {"env": "${TEST_VAR}"}}
            original_copy = original.copy()
            
            result = resolve_env_vars(original)
            
            # Result should have resolved variables
            assert result["key"] == "test_value"
            assert result["nested"]["env"] == "test_value"
            
            # Original should be unchanged
            assert original == original_copy
        
        # Test with deep_merge
        dict1 = {"a": 1, "b": {"c": 2}}
        dict1_copy = {"a": 1, "b": {"c": 2}}
        dict2 = {"b": {"d": 3}, "e": 4}
        dict2_copy = {"b": {"d": 3}, "e": 4}
        
        result = deep_merge(dict1, dict2)
        
        # Original dictionaries should be unchanged
        assert dict1 == dict1_copy
        assert dict2 == dict2_copy
    
    def test_transformer_extensibility(self):
        """Test that the transformer system is extensible."""
        # Should be able to add custom transformations easily
        
        # Create a custom transformation
        def uppercase_keys(config: Dict[str, Any]) -> Dict[str, Any]:
            """Transform all string values to uppercase."""
            result = {}
            for key, value in config.items():
                if isinstance(value, str):
                    result[key] = value.upper()
                elif isinstance(value, dict):
                    result[key] = uppercase_keys(value)
                else:
                    result[key] = value
            return result
        
        # Create a transformer with the custom transformation
        transformer = ConfigTransformer()
        transformer.add_transformation(uppercase_keys)
        
        # Apply the transformation
        result = transformer.transform({
            "string": "value",
            "number": 42,
            "nested": {"key": "nested_value"}
        })
        
        # String values should be uppercase
        assert result["string"] == "VALUE"
        # Non-string values should be unchanged
        assert result["number"] == 42
        # Nested values should also be transformed
        assert result["nested"]["key"] == "NESTED_VALUE"
    
    def test_transformer_error_handling(self):
        """Test that transformers handle errors appropriately."""
        # Create a transformation that can fail
        def failing_transform(config: Dict[str, Any]) -> Dict[str, Any]:
            if "trigger_error" in config:
                raise ValueError("Transformation error")
            return config
        
        # Create a transformer with the failing transformation
        transformer = ConfigTransformer()
        transformer.add_transformation(failing_transform)
        
        # Normal case should work
        assert transformer.transform({"normal": "value"}) == {"normal": "value"}
        
        # Error case should propagate the error
        with pytest.raises(ValueError, match="Transformation error"):
            transformer.transform({"trigger_error": True})
    
    def test_transformer_dependency_inversion(self):
        """Test that transformers follow the Dependency Inversion Principle."""
        # Transformer should depend on abstractions, not details
        
        # Create a mock transformation function
        mock_transform = mock.Mock(side_effect=lambda x: {**x, "transformed": True})
        
        # Create a transformer with the mock
        transformer = ConfigTransformer()
        transformer.add_transformation(mock_transform)
        
        # Apply the transformation
        result = transformer.transform({"original": True})
        
        # Mock should be called with the config
        mock_transform.assert_called_once_with({"original": True})
        
        # Result should include the transformation
        assert result == {"original": True, "transformed": True}


class TestResolveEnvVars:
    """Tests for the resolve_env_vars transformation."""
    
    def test_basic_resolution(self):
        """Test basic environment variable resolution."""
        with mock.patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            config = {"key": "${TEST_VAR}"}
            result = resolve_env_vars(config)
            assert result["key"] == "test_value"
    
    def test_nested_resolution(self):
        """Test resolution in nested dictionaries."""
        with mock.patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            config = {"nested": {"key": "${TEST_VAR}"}}
            result = resolve_env_vars(config)
            assert result["nested"]["key"] == "test_value"
    
    def test_partial_resolution(self):
        """Test resolution in strings with other content."""
        with mock.patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            config = {"key": "prefix_${TEST_VAR}_suffix"}
            result = resolve_env_vars(config)
            assert result["key"] == "prefix_test_value_suffix"
    
    def test_missing_env_var(self):
        """Test behavior with missing environment variables."""
        # Remove the variable if it exists
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]
            
        config = {"key": "${NONEXISTENT_VAR}"}
        result = resolve_env_vars(config)
        
        # Should keep the placeholder if variable doesn't exist
        assert result["key"] == "${NONEXISTENT_VAR}"
    
    def test_multiple_env_vars(self):
        """Test resolution with multiple variables in one string."""
        with mock.patch.dict(os.environ, {
            "VAR1": "value1",
            "VAR2": "value2"
        }):
            config = {"key": "${VAR1}_${VAR2}"}
            result = resolve_env_vars(config)
            assert result["key"] == "value1_value2"


class TestDeepMerge:
    """Tests for the deep_merge transformation."""
    
    def test_basic_merge(self):
        """Test basic dictionary merging."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        result = deep_merge(dict1, dict2)
        
        # Later values should override earlier ones
        assert result == {"a": 1, "b": 3, "c": 4}
        
        # Original dicts should be unchanged
        assert dict1 == {"a": 1, "b": 2}
        assert dict2 == {"b": 3, "c": 4}
    
    def test_nested_merge(self):
        """Test merging with nested dictionaries."""
        dict1 = {"a": 1, "nested": {"x": 1, "y": 2}}
        dict2 = {"b": 2, "nested": {"y": 3, "z": 4}}
        result = deep_merge(dict1, dict2)
        
        # Nested dictionaries should be merged recursively
        assert result == {
            "a": 1, 
            "b": 2, 
            "nested": {"x": 1, "y": 3, "z": 4}
        }
    
    def test_merge_with_non_dicts(self):
        """Test merging behavior with non-dictionary values."""
        dict1 = {"a": 1, "b": {"x": 1}}
        dict2 = {"b": 2}  # Non-dict value overrides dict
        result = deep_merge(dict1, dict2)
        
        # Non-dict should override dict
        assert result == {"a": 1, "b": 2}
        
        # Test the reverse
        dict3 = {"a": 1, "b": 2}
        dict4 = {"b": {"x": 1}}  # Dict overrides non-dict
        result = deep_merge(dict3, dict4)
        
        # Dict should override non-dict
        assert result == {"a": 1, "b": {"x": 1}}
    
    def test_merge_with_lists(self):
        """Test merging behavior with list values."""
        dict1 = {"a": 1, "list": [1, 2]}
        dict2 = {"b": 2, "list": [3, 4]}
        result = deep_merge(dict1, dict2)
        
        # Lists should be replaced, not merged
        assert result == {"a": 1, "b": 2, "list": [3, 4]}


class TestAddDefaultValues:
    """Tests for the add_default_values transformation."""
    
    def test_basic_defaults(self):
        """Test adding basic default values."""
        config = {"existing": "value"}
        defaults = {"existing": "default", "missing": "default"}
        
        result = add_default_values(config, defaults)
        
        # Existing values should not be overridden
        assert result["existing"] == "value"
        # Missing values should be added
        assert result["missing"] == "default"
    
    def test_nested_defaults(self):
        """Test adding nested default values."""
        config = {
            "level1": {
                "existing": "value"
            }
        }
        defaults = {
            "level1": {
                "existing": "default",
                "missing": "default"
            },
            "level2": {
                "key": "default"
            }
        }
        
        result = add_default_values(config, defaults)
        
        # Existing nested values should not be overridden
        assert result["level1"]["existing"] == "value"
        # Missing nested values should be added
        assert result["level1"]["missing"] == "default"
        # Missing nested dictionaries should be added
        assert result["level2"]["key"] == "default"
    
    def test_empty_config(self):
        """Test with an empty configuration."""
        config = {}
        defaults = {"key1": "value1", "key2": "value2"}
        
        result = add_default_values(config, defaults)
        
        # All defaults should be added
        assert result == defaults
        # But should be a different object
        assert result is not defaults