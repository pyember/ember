"""Tests for the Structural JIT strategy."""

import pytest
from typing import Dict, Any

from ember.xcs.jit.strategies.structural import StructuralStrategy


# Simple test function for JIT compilation
def simple_function(*, inputs):
    return {"result": inputs["value"] * 2}


# Simple container-like operator for structural compilation
class ContainerOperator:
    def __init__(self):
        self.op1 = NestedOperator()
        self.op2 = NestedOperator()
    
    def forward(self, *, inputs):
        intermediate = self.op1(inputs=inputs)
        # Make sure to pass the full intermediate result, not just a part of it
        return self.op2(inputs=intermediate)
    
    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)


class NestedOperator:
    def __init__(self):
        pass
    
    def forward(self, *, inputs):
        # Check whether we're getting 'value' or 'result' as input
        if "result" in inputs:
            return {"result": inputs["result"] * 2}
        else:
            return {"result": inputs["value"] * 2}
    
    def __call__(self, *, inputs):
        return self.forward(inputs=inputs)


# Simple operator with specification
class OperatorWithSpecification:
    specification = {"type": "test"}
    
    def forward(self, *, inputs):
        return {"result": inputs["value"] * 3}


def test_structural_strategy_analyze_function():
    """Test analysis of a simple function with structural strategy."""
    strategy = StructuralStrategy()
    
    # Analyze simple function
    analysis = strategy.analyze(simple_function)
    
    # Simple functions should get a low score for structural strategy
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    assert analysis["score"] < 30  # Simple functions aren't good for structural JIT


def test_structural_strategy_analyze_container():
    """Test analysis of a container operator with structural strategy."""
    strategy = StructuralStrategy()
    
    # Analyze container operator
    analysis = strategy.analyze(ContainerOperator)
    
    # Container operators should get a high score for structural strategy
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    assert analysis["score"] >= 30  # Container operators are good for structural JIT
    
    # Check features of container operator
    features = analysis["features"]
    assert features["is_class"] is True
    assert features["has_forward"] is True


def test_structural_strategy_analyze_with_specification():
    """Test analysis of an operator with specification."""
    strategy = StructuralStrategy()
    
    # Analyze operator with specification
    analysis = strategy.analyze(OperatorWithSpecification)
    
    # Operators with specification should get points for that
    assert "score" in analysis
    assert "rationale" in analysis
    assert "features" in analysis
    assert analysis["score"] >= 10  # Having specification gives at least 10 points
    
    # Check features
    features = analysis["features"]
    assert features["has_specification"] is True


def test_structural_strategy_compile():
    """Test compilation of a function with structural strategy."""
    strategy = StructuralStrategy()
    
    # Compile simple function
    compiled_func = strategy.compile(simple_function)
    
    # Test the compiled function with some input
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
    assert compiled_func._jit_strategy == "structural"


def test_structural_strategy_compile_container():
    """Test compilation of a container operator with structural strategy."""
    strategy = StructuralStrategy()
    
    # Create a container operator instance
    container = ContainerOperator()
    
    # Compile the container operator
    compiled_container = strategy.compile(container)
    
    # Test the compiled container with some input
    result = compiled_container(inputs={"value": 5})
    
    # Since the container calls op1 and op2 in sequence, each multiplying by 2,
    # the final result should be value * 2 * 2 = value * 4
    assert result["result"] == 20
    
    # Check that control methods were added
    assert hasattr(compiled_container, "disable_jit")
    assert hasattr(compiled_container, "enable_jit")
    assert hasattr(compiled_container, "get_stats")


def test_structural_strategy_disable_jit():
    """Test disabling JIT for a compiled function."""
    strategy = StructuralStrategy()
    
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


def test_structural_strategy_execute_with_fallback():
    """Test fallback execution when graph execution fails."""
    import unittest.mock
    from ember.xcs.jit import execution_utils
    
    # Monkey patch execute_compiled_graph to raise an exception
    original_func = execution_utils.execute_compiled_graph
    
    try:
        # Replace execute_compiled_graph with a function that raises an exception
        execution_utils.execute_compiled_graph = unittest.mock.Mock(side_effect=Exception("Mock error"))
        
        strategy = StructuralStrategy()
        
        # Create a mock graph
        mock_graph = object()  # Any object will do
        
        # Get a cache to use
        from ember.xcs.jit.cache import get_cache
        cache = get_cache()
        
        # Execute with fallback - should trigger the exception path
        result = strategy.execute_with_fallback(
            mock_graph, 
            simple_function, 
            {"value": 5}, 
            cache
        )
        
        # Check that fallback execution gave the correct result
        assert result["result"] == 10
        
        # Test with a more complex function
        def complex_func(*, inputs):
            return {"result": inputs["value"] * 3 + 1}
        
        # Execute with fallback using the complex function
        result = strategy.execute_with_fallback(
            mock_graph,
            complex_func,
            {"value": 3},
            cache
        )
        
        # Check that fallback execution worked correctly
        assert result["result"] == 10  # 3 * 3 + 1 = 10
        
        # Verify that mock was called
        assert execution_utils.execute_compiled_graph.call_count >= 2
    
    finally:
        # Restore the original function
        execution_utils.execute_compiled_graph = original_func