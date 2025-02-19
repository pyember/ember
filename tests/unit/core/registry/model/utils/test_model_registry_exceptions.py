#!/usr/bin/env python3
"""Unit tests for model registry exceptions.
"""

import pytest

from src.ember.core.registry.model.utils.model_registry_exceptions import (
    ModelRegistrationError,
    ModelDiscoveryError,
)


def test_model_registration_error() -> None:
    """Test that ModelRegistrationError contains correct error message."""
    error = ModelRegistrationError("TestModel", "Some reason")
    assert "TestModel" in str(error)
    assert "Some reason" in str(error)


def test_model_discovery_error() -> None:
    """Test that ModelDiscoveryError contains correct provider and reason."""
    error = ModelDiscoveryError("TestProvider", "Discovery failed")
    assert "TestProvider" in str(error)
    assert "Discovery failed" in str(error)