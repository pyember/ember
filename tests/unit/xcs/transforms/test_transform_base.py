"""Unit tests for the XCS transform base classes and protocols.

Tests the foundation for all XCS transformations, including common utilities
for batching, partitioning, and result handling.
"""

import pytest

from ember.xcs import (
    BaseTransformation,
    BatchingOptions,
    ParallelOptions,
    TransformError,
    compose
)
from ember.xcs.transforms.transform_base import (
    get_batch_size,
    split_batch,
    combine_outputs
)


class TestTransformBase:
    """Test suite for transform base functionality."""

    def test_transform_error_creation(self) -> None:
        """Test creation of transformation errors."""
        # Test basic error creation
        error = TransformError("Basic error")
        assert str(error) == "Basic error"
        
        # Test with transform context
        error = TransformError.for_transform("vmap", "Batch size mismatch")
        assert "[vmap]" in str(error)
        assert "Batch size mismatch" in str(error)
        
        # Test with details
        error = TransformError.for_transform(
            "pmap", 
            "Worker failure", 
            details={"worker_id": 3}
        )
        assert "pmap" in str(error)
        assert "Worker failure" in str(error)
        assert "details" in str(error).lower()
        
        # Test with cause
        cause = ValueError("Original error")
        error = TransformError.for_transform("vmap", "Wrapper", cause=cause)
        assert error.__cause__ == cause


class TestBatchingOptions:
    """Test suite for batching options validation."""
    
    def test_valid_options(self) -> None:
        """Test valid batching options configurations."""
        # Test with default values
        options = BatchingOptions()
        options.validate()  # Should not raise
        
        # Test with specific in_axes
        options = BatchingOptions(in_axes={"x": 0, "y": 1})
        options.validate()  # Should not raise
        
        # Test with parallel batching
        options = BatchingOptions(
            in_axes=0, 
            out_axis=0, 
            batch_size=10, 
            parallel=True, 
            max_workers=4
        )
        options.validate()  # Should not raise
    
    def test_invalid_options(self) -> None:
        """Test validation of invalid configurations."""
        # Test invalid in_axes type
        with pytest.raises(TransformError, match="in_axes must be"):
            options = BatchingOptions(in_axes=["invalid"])
            options.validate()
        
        # Test invalid batch_size
        with pytest.raises(TransformError, match="batch_size must be positive"):
            options = BatchingOptions(batch_size=-5)
            options.validate()
        
        # Test invalid max_workers
        with pytest.raises(TransformError, match="max_workers must be positive"):
            options = BatchingOptions(max_workers=0)
            options.validate()


class TestParallelOptions:
    """Test suite for parallel execution options validation."""
    
    def test_valid_options(self) -> None:
        """Test valid parallel options configurations."""
        # Test with default values
        options = ParallelOptions()
        options.validate()  # Should not raise
        
        # Test with all parameters
        options = ParallelOptions(
            num_workers=4,
            continue_on_errors=True,
            timeout_seconds=30.0,
            return_partial=True
        )
        options.validate()  # Should not raise
    
    def test_invalid_options(self) -> None:
        """Test validation of invalid configurations."""
        # Test invalid num_workers
        with pytest.raises(TransformError, match="num_workers must be positive"):
            options = ParallelOptions(num_workers=-2)
            options.validate()
        
        # Test invalid timeout
        with pytest.raises(TransformError, match="timeout_seconds must be positive"):
            options = ParallelOptions(timeout_seconds=0)
            options.validate()


class TestBaseTransformation:
    """Test suite for the BaseTransformation class."""
    
    def test_metadata_preservation(self) -> None:
        """Test function metadata preservation during transformation."""
        # Create a sample transformation
        class TestTransform(BaseTransformation):
            def __call__(self, fn):
                transformed = lambda *args, **kwargs: fn(*args, **kwargs)
                return self._preserve_function_metadata(fn, transformed)
        
        # Sample function with docstring and metadata
        def sample_fn(x: int, y: int) -> int:
            """Sample function that adds two numbers."""
            return x + y
        
        # Apply transformation
        transform = TestTransform(name="test_transform")
        transformed_fn = transform(sample_fn)
        
        # Check metadata preservation
        assert transformed_fn.__name__ == "sample_fn"
        assert transformed_fn.__doc__ == sample_fn.__doc__
        assert hasattr(transformed_fn, "_original_function")
        assert transformed_fn._original_function == sample_fn
        assert hasattr(transformed_fn, "_test_transform_transform")


class TestBatchUtilities:
    """Test suite for batching utility functions."""
    
    def test_get_batch_size(self) -> None:
        """Test batch size determination from inputs."""
        # Test with simple in_axes
        inputs = {"x": [1, 2, 3], "y": [4, 5, 6]}
        size = get_batch_size(inputs, in_axes=0)
        assert size == 3
        
        # Test with dict in_axes
        inputs = {"x": [1, 2, 3], "y": "non-batched"}
        size = get_batch_size(inputs, in_axes={"x": 0})
        assert size == 3
        
        # Test with inconsistent batch sizes
        inputs = {"x": [1, 2, 3], "y": [4, 5]}
        with pytest.raises(TransformError, match="Inconsistent batch"):
            get_batch_size(inputs, in_axes=0)
        
        # Test with empty input
        inputs = {"x": []}
        with pytest.raises(TransformError, match="Empty batch"):
            get_batch_size(inputs, in_axes=0)
    
    def test_split_batch(self) -> None:
        """Test extraction of elements from batched inputs."""
        # Test with simple in_axes
        inputs = {"x": [1, 2, 3], "y": [4, 5, 6]}
        element = split_batch(inputs, in_axes=0, index=1)
        assert element == {"x": 2, "y": 5}
        
        # Test with dict in_axes
        inputs = {"x": [1, 2, 3], "y": "non-batched"}
        element = split_batch(inputs, in_axes={"x": 0}, index=1)
        assert element == {"x": 2, "y": "non-batched"}
    
    def test_combine_outputs(self) -> None:
        """Test combining individual results into batched output."""
        # Test basic combination
        outputs = [
            {"result": 1, "meta": "a"},
            {"result": 2, "meta": "b"},
            {"result": 3, "meta": "c"}
        ]
        combined = combine_outputs(outputs)
        assert combined["result"] == [1, 2, 3]
        assert combined["meta"] == ["a", "b", "c"]
        
        # Test with empty list
        assert combine_outputs([]) == {}
        
        # Test with non-dict results
        assert "results" in combine_outputs([1, 2, 3])


class TestComposeFunction:
    """Test suite for the compose utility function."""
    
    def test_compose_transformations(self) -> None:
        """Test composition of multiple transformations."""
        # Create sample transformations
        class AddValue(BaseTransformation):
            def __init__(self, value: int):
                super().__init__("add_value")
                self.value = value
                
            def __call__(self, fn):
                def wrapped(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    return result + self.value
                return self._preserve_function_metadata(fn, wrapped)
        
        class MultiplyValue(BaseTransformation):
            def __init__(self, value: int):
                super().__init__("multiply_value")
                self.value = value
                
            def __call__(self, fn):
                def wrapped(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    return result * self.value
                return self._preserve_function_metadata(fn, wrapped)
        
        # Function to transform
        def identity(x: int) -> int:
            return x
        
        # Apply transformations
        add5 = AddValue(5)
        multiply2 = MultiplyValue(2)
        
        # Test individual transforms
        assert add5(identity)(10) == 15
        assert multiply2(identity)(10) == 20
        
        # Test composed transforms in different orders
        composed1 = compose(add5, multiply2)
        composed2 = compose(multiply2, add5)
        
        # Should apply right-to-left (like function composition)
        # composed1 = add5(multiply2(identity)) = (x*2) + 5
        # composed2 = multiply2(add5(identity)) = (x+5) * 2
        assert composed1(identity)(10) == 25  # (10*2) + 5 = 25
        assert composed2(identity)(10) == 30  # (10+5) * 2 = 30