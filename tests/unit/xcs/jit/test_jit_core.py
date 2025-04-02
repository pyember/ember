"""Tests for the core JIT functionality."""

import pytest
from typing import Dict, Any

from ember.xcs.jit import jit, JITMode
from ember.xcs.jit.core import JITSettings


def test_jit_settings_init():
    """Test initialization of JIT settings."""
    settings = JITSettings(
        mode=JITMode.AUTO,
        force_trace=False,
        sample_input={"test": "value"},
        recursive=True
    )
    
    assert settings.mode == JITMode.AUTO
    assert settings.force_trace is False
    assert settings.sample_input == {"test": "value"}
    assert settings.recursive is True


def test_string_mode_conversion():
    """Test conversion of string modes to enum."""
    settings = JITSettings(mode="trace")
    assert settings.mode == JITMode.TRACE
    
    settings = JITSettings(mode="structural")
    assert settings.mode == JITMode.STRUCTURAL
    
    settings = JITSettings(mode="enhanced")
    assert settings.mode == JITMode.ENHANCED


def test_jit_decorator():
    """Test basic JIT decorator functionality."""
    @jit
    def simple_function(*, inputs):
        return {"result": inputs["value"] * 2}
    
    result = simple_function(inputs={"value": 5})
    assert result["result"] == 10


def test_jit_with_explicit_mode():
    """Test JIT with explicit mode specification."""
    @jit(mode=JITMode.TRACE)
    def trace_function(*, inputs):
        return {"result": inputs["value"] * 2}
    
    result = trace_function(inputs={"value": 5})
    assert result["result"] == 10
    
    @jit(mode="structural")
    def structural_function(*, inputs):
        return {"result": inputs["value"] * 3}
    
    result = structural_function(inputs={"value": 5})
    assert result["result"] == 15