"""Tests for JIT strategy selection functionality."""

import pytest
from typing import Dict, Any

from ember.xcs.jit import JITMode
from ember.xcs.jit.core import jit, explain_jit_selection, StrategySelector


# Simple test function (should favor Trace strategy)
def simple_function(*, inputs):
    return {"result": inputs["value"] * 2}


# Simple operator class with correct input handling for nested operators
class SimpleOperator:
    def forward(self, *, inputs):
        value = inputs.get("value", inputs.get("result", 0))
        return {"result": value * 2}
    
    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)


# Container operator with proper input handling
class ContainerOperator:
    def __init__(self):
        self.op1 = SimpleOperator()
        self.op2 = SimpleOperator()
    
    def forward(self, *, inputs):
        intermediate = self.op1(inputs=inputs)
        return self.op2(inputs=intermediate)
    
    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)


# Ensemble operator (should favor Enhanced strategy)
class EnsembleOperator:
    def __init__(self):
        # Give it "ensemble" in the name for the enhanced strategy to recognize
        self.ensemble_name = "ensemble_op"
        self.models = [SimpleOperator() for _ in range(3)]
    
    def forward(self, *, inputs):
        results = []
        for model in self.models:
            results.append(model(inputs=inputs))
        
        # Aggregate results (sum)
        total = sum(r["result"] for r in results)
        return {"result": total / len(results)}
    
    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)
    
    def __iter__(self):
        """Add iteration capability for enhanced strategy to detect."""
        return iter(self.models)


def test_explain_jit_selection():
    """Test that explain_jit_selection provides strategy analysis."""
    # Test with simple function
    analysis = explain_jit_selection(simple_function)
    
    # Should have analysis from all three strategies
    assert "trace" in analysis
    assert "structural" in analysis
    assert "enhanced" in analysis
    
    # Each analysis should have a score and rationale
    for strategy_name, strategy_analysis in analysis.items():
        assert "score" in strategy_analysis
        assert "rationale" in strategy_analysis
    
    # Check that trace gets a higher score for simple functions than enhanced
    assert analysis["trace"]["score"] >= analysis["enhanced"]["score"]


def test_jit_decorator_with_auto_mode():
    """Test JIT decorator with auto mode selects appropriate strategy."""
    # Use the jit decorator with auto mode on simple function
    @jit(mode=JITMode.AUTO)
    def auto_function(*, inputs):
        return {"result": inputs["value"] * 2}
    
    # Execute the decorated function
    result = auto_function(inputs={"value": 5})
    assert result["result"] == 10
    
    # Use jit decorator with string mode
    @jit(mode="auto")
    def string_auto_function(*, inputs):
        return {"result": inputs["value"] * 3}
    
    # Execute the decorated function
    result = string_auto_function(inputs={"value": 5})
    assert result["result"] == 15


def test_jit_with_forced_strategy():
    """Test JIT with explicitly forced strategy."""
    # Force trace strategy
    @jit(mode=JITMode.TRACE)
    def trace_function(*, inputs):
        return {"result": inputs["value"] * 2}
    
    # Execute and check that it works
    result = trace_function(inputs={"value": 5})
    assert result["result"] == 10
    
    # Force structural strategy
    @jit(mode=JITMode.STRUCTURAL)
    def structural_func(*, inputs):
        return {"result": inputs["value"] * 2}
    
    # Execute and check that it works
    result = structural_func(inputs={"value": 5})
    assert result["result"] == 10
    
    # Force enhanced strategy
    @jit(mode=JITMode.ENHANCED)
    def enhanced_func(*, inputs):
        return {"result": inputs["value"] * 2}
    
    # Execute and check that it works
    result = enhanced_func(inputs={"value": 5})
    assert result["result"] == 10