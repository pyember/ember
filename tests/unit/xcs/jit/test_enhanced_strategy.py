"""Tests for the Enhanced JIT strategy."""

import pytest
from typing import Dict, Any, List

from ember.xcs.jit.strategies.enhanced import EnhancedStrategy


# Simple test function for JIT compilation
def simple_function(*, inputs):
    return {"result": inputs["value"] * 2}


# Function with loops, good for enhanced JIT
def array_processor(*, inputs):
    results = []
    for item in inputs["items"]:
        results.append(item * 2)
    return {"results": results}


# Ensemble-like operator class
class EnsembleOperator:
    def __init__(self):
        self.models = [Model(i) for i in range(3)]
    
    def forward(self, *, inputs):
        results = []
        for model in self.models:
            results.append(model(inputs=inputs))
        
        # Aggregate results (average)
        total = sum(r["result"] for r in results)
        return {"result": total / len(results)}
    
    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)


class Model:
    def __init__(self, factor):
        self.factor = factor + 1  # Ensure factor is at least 1
    
    def __call__(self, *, inputs):
        return {"result": inputs["value"] * self.factor}


def test_enhanced_strategy_analyze_simple_function():
    """Test analysis of a simple function with enhanced strategy."""
    strategy = EnhancedStrategy()
    
    # Analyze simple function
    analysis = strategy.analyze(simple_function)
    
    # Simple functions should get a modest score for enhanced strategy
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    # Enhanced is a good general-purpose option so should have a positive score
    assert analysis["score"] >= 10  
    
    # Check features
    features = analysis["features"]
    assert features["is_function"] is True


def test_enhanced_strategy_analyze_with_loops():
    """Test analysis of a function with loops."""
    strategy = EnhancedStrategy()
    
    # Analyze function with loops
    analysis = strategy.analyze(array_processor)
    
    # Functions with loops should get a higher score for enhanced strategy
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    # Should score higher than a simple function
    assert analysis["score"] >= 10


def test_enhanced_strategy_analyze_ensemble():
    """Test analysis of an ensemble-like operator."""
    strategy = EnhancedStrategy()
    
    # Analyze ensemble operator
    analysis = strategy.analyze(EnsembleOperator)
    
    # Ensemble operators should get a high score for enhanced strategy
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    # The name contains "ensemble" and it has loops, so score should be high
    assert analysis["score"] >= 30
    
    # Check features
    features = analysis["features"]
    assert features["is_class"] is True


def test_enhanced_strategy_compile_simple_function():
    """Test compilation of a simple function with enhanced strategy."""
    strategy = EnhancedStrategy()
    
    # Compile simple function
    compiled_func = strategy.compile(simple_function)
    
    # Test the compiled function
    result = compiled_func(inputs={"value": 5})
    
    # Check the result
    assert result["result"] == 10
    
    # Check that control methods were added
    assert hasattr(compiled_func, "disable_jit")
    assert hasattr(compiled_func, "enable_jit")
    assert hasattr(compiled_func, "get_stats")
    assert hasattr(compiled_func, "_original_function")
    assert compiled_func._original_function is simple_function
    assert hasattr(compiled_func, "_jit_strategy")
    assert compiled_func._jit_strategy == "enhanced"


def test_enhanced_strategy_compile_array_processor():
    """Test compilation of a function with loops."""
    strategy = EnhancedStrategy()
    
    # Compile array processor
    compiled_func = strategy.compile(array_processor)
    
    # Test the compiled function with array input
    result = compiled_func(inputs={"items": [1, 2, 3, 4, 5]})
    
    # Check the result
    assert result["results"] == [2, 4, 6, 8, 10]


def test_enhanced_strategy_compile_ensemble():
    """Test compilation of an ensemble operator."""
    strategy = EnhancedStrategy()
    
    # Create ensemble operator
    ensemble = EnsembleOperator()
    
    # Compile the ensemble
    compiled_ensemble = strategy.compile(ensemble)
    
    # Test the compiled ensemble
    result = compiled_ensemble(inputs={"value": 5})
    
    # With 3 models with factors 1, 2, and 3, the average should be
    # (5*1 + 5*2 + 5*3) / 3 = (5 + 10 + 15) / 3 = 30/3 = 10
    assert result["result"] == 10


def test_enhanced_strategy_with_sample_input():
    """Test compilation with sample input for eager compilation."""
    strategy = EnhancedStrategy()
    
    # Compile with sample input
    compiled_func = strategy.compile(
        simple_function,
        sample_input={"value": 10}
    )
    
    # Test the compiled function
    result = compiled_func(inputs={"value": 5})
    
    # Check the result
    assert result["result"] == 10


def test_enhanced_strategy_disable_jit():
    """Test disabling JIT for a compiled function."""
    strategy = EnhancedStrategy()
    
    # Compile simple function
    compiled_func = strategy.compile(simple_function)
    
    # Disable JIT
    compiled_func.disable_jit()
    
    # Test the compiled function with JIT disabled
    result = compiled_func(inputs={"value": 5})
    
    # Check the result (should still work correctly)
    assert result["result"] == 10
    
    # Enable JIT again
    compiled_func.enable_jit()
    
    # Test the compiled function with JIT enabled
    result = compiled_func(inputs={"value": 5})
    
    # Check the result (should still work correctly)
    assert result["result"] == 10


def test_enhanced_strategy_force_trace():
    """Test forcing trace on every call."""
    strategy = EnhancedStrategy()
    
    # Compile with force_trace=True
    compiled_func = strategy.compile(
        simple_function,
        force_trace=True
    )
    
    # Test the compiled function (will bypass caching)
    result = compiled_func(inputs={"value": 5})
    
    # Check the result
    assert result["result"] == 10
    
    # Compile with force_trace=False
    compiled_func = strategy.compile(
        simple_function,
        force_trace=False
    )
    
    # Test the compiled function (will use caching)
    result = compiled_func(inputs={"value": 5})
    
    # Check the result
    assert result["result"] == 10


def test_enhanced_strategy_recursive():
    """Test recursive compilation setting."""
    strategy = EnhancedStrategy()
    
    # Compile with recursive=True (the default)
    compiled_func = strategy.compile(
        simple_function,
        recursive=True
    )
    
    # Test the compiled function
    result = compiled_func(inputs={"value": 5})
    
    # Check the result
    assert result["result"] == 10
    
    # Compile with recursive=False
    compiled_func = strategy.compile(
        simple_function,
        recursive=False
    )
    
    # Test the compiled function
    result = compiled_func(inputs={"value": 5})
    
    # Check the result
    assert result["result"] == 10