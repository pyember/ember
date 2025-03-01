#!/usr/bin/env python3
"""Unit tests for model registry exceptions.
"""

import pytest

from ember.core.registry.model.base.utils.model_registry_exceptions import (
    ModelRegistrationError,
    ModelDiscoveryError,
)


def test_model_registration_error() -> None:
    """Test that ModelRegistrationError contains correct error message."""
    error = ModelRegistrationError(model_name="TestModel", reason="Some reason")
    assert "TestModel" in str(error)
    assert "Some reason" in str(error)


def test_model_discovery_error() -> None:
    """Test that ModelDiscoveryError contains correct provider and reason."""
    error = ModelDiscoveryError(provider="TestProvider", reason="Discovery failed")
    assert "TestProvider" in str(error)
    assert "Discovery failed" in str(error)
