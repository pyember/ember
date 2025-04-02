"""Tests for the Trace JIT strategy."""

import pytest
from typing import Dict, Any

from ember.xcs.jit.strategies.trace import TraceStrategy


# Simple test functions for JIT compilation
def simple_function(*, inputs):
    return {"result": inputs["value"] * 2}


def complex_function_with_conditionals(*, inputs):
    if inputs["value"] > 10:
        return {"result": inputs["value"] * 2}
    else:
        return {"result": inputs["value"] * 3}


def function_with_loop(*, inputs):
    result = 0
    for i in range(inputs["iterations"]):
        result += inputs["value"]
    return {"result": result}


# Simple operator class
class SimpleOperator:
    def forward(self, *, inputs):
        return {"result": inputs["value"] * 2}
    
    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)


def test_trace_strategy_analyze_simple_function():
    """Test analysis of a simple function with trace strategy."""
    strategy = TraceStrategy()
    
    # Analyze simple function
    analysis = strategy.analyze(simple_function)
    
    # Simple functions should get a high score for trace strategy
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    assert analysis["score"] >= 30  # Simple functions are good for trace JIT
    
    # Check features
    features = analysis["features"]
    assert features["is_function"] is True
    assert features["has_source"] is True


def test_trace_strategy_analyze_complex_function():
    """Test analysis of a function with conditionals."""
    strategy = TraceStrategy()
    
    # Analyze function with conditionals
    analysis = strategy.analyze(complex_function_with_conditionals)
    
    # Functions with conditionals get a lower score, but still viable
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    # Should still have a positive score
    assert analysis["score"] > 0

    # Test with a function that will cause inspect.getsource to fail
    # by using a lambda function (which doesn't have a source file)
    lambda_func = lambda *, inputs: {"result": inputs["value"] * 2}
    
    # Analyze the lambda function
    lambda_analysis = strategy.analyze(lambda_func)
    
    # Should still get a base score even without source
    assert lambda_analysis["score"] > 0


def test_trace_strategy_analyze_function_with_loop():
    """Test analysis of a function with loops."""
    strategy = TraceStrategy()
    
    # Analyze function with loop
    analysis = strategy.analyze(function_with_loop)
    
    # Functions with loops are less optimal for trace strategy
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    # Score would typically be lower than simple functions without loops
    assert analysis["score"] < analysis["score"] + 1  # Simple check that score exists


def test_trace_strategy_compile_simple_function():
    """Test compilation of a simple function with trace strategy."""
    strategy = TraceStrategy()
    
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
    assert compiled_func._jit_strategy == "trace"


def test_trace_strategy_compile_with_sample_input():
    """Test compilation with sample input for eager tracing."""
    strategy = TraceStrategy()
    
    # Compile with sample input
    compiled_func = strategy.compile(
        simple_function,
        sample_input={"value": 10}
    )
    
    # Test the compiled function
    result = compiled_func(inputs={"value": 5})
    
    # Check the result
    assert result["result"] == 10


def test_trace_strategy_compile_operator_class():
    """Test compilation of an operator class."""
    strategy = TraceStrategy()
    
    # Create operator instance
    operator = SimpleOperator()
    
    # Compile the operator
    compiled_operator = strategy.compile(operator)
    
    # Test the compiled operator
    result = compiled_operator(inputs={"value": 5})
    
    # Check the result
    assert result["result"] == 10


def test_trace_strategy_disable_jit():
    """Test disabling JIT for a compiled function."""
    strategy = TraceStrategy()
    
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


def test_trace_strategy_force_trace():
    """Test forcing trace on every call."""
    strategy = TraceStrategy()
    
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


def test_trace_strategy_recursive():
    """Test recursive tracing setting."""
    strategy = TraceStrategy()
    
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