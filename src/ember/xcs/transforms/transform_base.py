"""Transform base classes and protocols.

Provides the foundation for all XCS transformations with a consistent interface
and common utilities. This module defines the core abstractions and shared
functionality for transforming functions and operators.

Key components:
1. TransformProtocol: Interface that all transformations must implement
2. BaseTransformation: Common foundation for all transformations
3. Utilities for batching, partitioning, and result aggregation
"""

import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Union, cast

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')
U = TypeVar('U')
InputsT = TypeVar('InputsT', bound=Dict[str, Any])
OutputsT = TypeVar('OutputsT', bound=Dict[str, Any])


class TransformError(Exception):
    """Base exception for all transformation errors."""
    
    @classmethod
    def for_transform(cls, transform_name: str, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None) -> "TransformError":
        """Create an error specific to a transformation.
        
        Args:
            transform_name: Name of the transformation where the error occurred
            message: Error message
            details: Additional details about the error
            cause: The original exception that caused this error
            
        Returns:
            A properly formatted transform error
        """
        formatted_message = f"[{transform_name}] {message}"
        if details:
            formatted_message += f" Details: {details}"
            
        error = cls(formatted_message)
        if cause:
            error.__cause__ = cause
            
        return error


class TransformProtocol(Protocol):
    """Protocol defining the interface for all transformations.
    
    All transformations must implement this protocol to ensure consistent
    behavior and interoperability.
    """
    
    def __call__(self, fn: Callable[..., T]) -> Callable[..., Any]:
        """Apply the transformation to a function.
        
        Args:
            fn: Function to transform
            
        Returns:
            Transformed function
        """
        ...


class BaseOptions:
    """Base class for all transformation options.
    
    Provides common functionality and validation for transformation options.
    """
    
    def validate(self) -> None:
        """Validate configuration settings.
        
        Checks that all settings have valid values and are internally consistent.
        
        Raises:
            TransformError: If validation fails
        """
        pass


class BatchingOptions(BaseOptions):
    """Configuration options for batch processing transforms.
    
    Controls batch dimension handling, batch sizes, and axis specifications
    for input and output batch dimensions.
    """
    
    def __init__(
        self,
        *,
        in_axes: Union[int, Dict[str, int]] = 0,
        out_axis: int = 0,
        batch_size: Optional[int] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        """Initialize batching options.
        
        Args:
            in_axes: Specification of which inputs are batched and on which axis.
                If an integer, applies to all inputs. If a dict, specifies axes
                for specific keys. Keys not specified are treated as non-batch inputs.
            out_axis: Axis in output for batched results
            batch_size: Optional maximum batch size for processing
            parallel: Whether to process batch elements in parallel
            max_workers: Maximum number of workers for parallel processing
        """
        self.in_axes = in_axes
        self.out_axis = out_axis
        self.batch_size = batch_size
        self.parallel = parallel
        self.max_workers = max_workers
        
    def validate(self) -> None:
        """Validate batching options.
        
        Ensures all options have valid values.
        
        Raises:
            TransformError: If validation fails
        """
        if not isinstance(self.in_axes, (int, dict)):
            raise TransformError(
                f"in_axes must be an int or dict, got {type(self.in_axes)}"
            )
            
        if self.batch_size is not None and self.batch_size <= 0:
            raise TransformError(
                f"batch_size must be positive, got {self.batch_size}"
            )
            
        if self.max_workers is not None and self.max_workers <= 0:
            raise TransformError(
                f"max_workers must be positive, got {self.max_workers}"
            )


class ParallelOptions(BaseOptions):
    """Configuration options for parallel execution transforms.
    
    Controls workers, task distribution, error handling, and timeout behavior
    for parallel execution.
    """
    
    def __init__(
        self,
        *,
        num_workers: Optional[int] = None,
        continue_on_errors: bool = False,
        timeout_seconds: Optional[float] = None,
        return_partial: bool = True,
    ) -> None:
        """Initialize parallel options.
        
        Args:
            num_workers: Number of worker threads to use
            continue_on_errors: Whether to continue execution if errors occur
            timeout_seconds: Maximum execution time before timeout
            return_partial: Whether to return partial results on timeout or error
        """
        self.num_workers = num_workers
        self.continue_on_errors = continue_on_errors
        self.timeout_seconds = timeout_seconds
        self.return_partial = return_partial
        
    def validate(self) -> None:
        """Validate parallel options.
        
        Ensures all options have valid values.
        
        Raises:
            TransformError: If validation fails
        """
        if self.num_workers is not None and self.num_workers <= 0:
            raise TransformError(
                f"num_workers must be positive, got {self.num_workers}"
            )
            
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise TransformError(
                f"timeout_seconds must be positive, got {self.timeout_seconds}"
            )


class BaseTransformation:
    """Base class for all transformations.
    
    Provides common functionality for transformations, including name tracking,
    validation, and decorator syntax support.
    """
    
    def __init__(self, name: str) -> None:
        """Initialize the transformation.
        
        Args:
            name: Transformation name for debugging and logging
        """
        self.name = name
    
    def __call__(self, fn: Callable[..., T]) -> Callable[..., Any]:
        """Apply the transformation to a function.
        
        Args:
            fn: Function to transform
            
        Returns:
            Transformed function
        """
        raise NotImplementedError(
            f"Transformation '{self.name}' does not implement __call__"
        )
        
    def _preserve_function_metadata(self, original_fn: Callable, transformed_fn: Callable) -> Callable:
        """Preserve function metadata from original to transformed function.
        
        Args:
            original_fn: Original function
            transformed_fn: Transformed function
            
        Returns:
            Transformed function with preserved metadata
        """
        functools.update_wrapper(transformed_fn, original_fn)
        # Add reference to the original function for introspection
        setattr(transformed_fn, "_original_function", original_fn)
        # Add reference to the transformation
        setattr(transformed_fn, f"_{self.name}_transform", True)
        
        return transformed_fn
    
    def _get_default_num_workers(self) -> int:
        """Determine default worker count based on system configuration.
        
        Returns:
            Optimal worker count for the current system
        """
        env_workers = os.environ.get("XCS_NUM_WORKERS")
        if env_workers:
            try:
                count = int(env_workers)
                if count > 0:
                    return count
            except ValueError:
                pass
                
        # Use CPU count - 1 by default to avoid overwhelming the system
        import multiprocessing
        return max(1, multiprocessing.cpu_count() - 1)


# Utility functions for batching

def get_batch_size(inputs: Dict[str, Any], in_axes: Union[int, Dict[str, int]]) -> int:
    """Determine the batch size from inputs.
    
    Args:
        inputs: Input dictionary
        in_axes: Input axis specification
        
    Returns:
        Determined batch size
        
    Raises:
        TransformError: If batch size cannot be determined or inputs are inconsistent
    """
    batch_size = None
    
    if isinstance(in_axes, int):
        # All inputs use the same axis
        for key, value in inputs.items():
            if not isinstance(value, (list, tuple)):
                continue
                
            if len(value) == 0:
                raise TransformError(f"Empty batch for input '{key}'")
                
            if batch_size is None:
                batch_size = len(value)
            elif batch_size != len(value):
                raise TransformError(
                    f"Inconsistent batch sizes: {batch_size} != {len(value)} for '{key}'"
                )
    else:
        # Different axes for different inputs
        for key, axis in in_axes.items():
            if key not in inputs:
                # This is allowed - batch spec may include optional inputs
                continue
                
            value = inputs[key]
            if not isinstance(value, (list, tuple)):
                raise TransformError(
                    f"Input '{key}' specified as batched but is not a sequence"
                )
                
            if len(value) == 0:
                raise TransformError(f"Empty batch for input '{key}'")
                
            if batch_size is None:
                batch_size = len(value)
            elif batch_size != len(value):
                raise TransformError(
                    f"Inconsistent batch sizes: {batch_size} != {len(value)} for '{key}'"
                )
    
    if batch_size is None:
        raise TransformError("Could not determine batch size from inputs")
        
    return batch_size


def split_batch(
    inputs: Dict[str, Any], 
    in_axes: Union[int, Dict[str, int]], 
    index: int
) -> Dict[str, Any]:
    """Extract a single element from a batch of inputs.
    
    Args:
        inputs: Batched input dictionary
        in_axes: Input axis specification
        index: Index to extract
        
    Returns:
        Dictionary with the extracted element for each batched input
    """
    result = {}
    
    if isinstance(in_axes, int):
        # All inputs use the same axis
        for key, value in inputs.items():
            if isinstance(value, (list, tuple)) and len(value) > index:
                result[key] = value[index]
            else:
                # Non-batched input, pass as-is
                result[key] = value
    else:
        # Different axes for different inputs
        for key, value in inputs.items():
            if key in in_axes and isinstance(value, (list, tuple)) and len(value) > index:
                result[key] = value[index]
            else:
                # Non-batched input, pass as-is
                result[key] = value
    
    return result


def combine_outputs(results: List[Dict[str, Any]], out_axis: int = 0) -> Dict[str, Any]:
    """Combine individual results into a batched output.
    
    Args:
        results: List of result dictionaries
        out_axis: Output axis for batching
        
    Returns:
        Combined dictionary with batched outputs
    """
    if not results:
        return {}
        
    combined = {}
    
    # Get all output keys
    keys = set()
    for result in results:
        if not isinstance(result, dict):
            # Handle non-dict results by wrapping them
            return {"results": results}
        keys.update(result.keys())
    
    # Check for scalar results with a "results" key for backward compatibility
    if len(results) == 1 and "results" in results[0] and not isinstance(results[0]["results"], list):
        return {"results": [results[0]["results"]]}
    
    # Combine each key
    for key in keys:
        values = [result.get(key) for result in results if key in result]
        
        if all(v is None for v in values):
            combined[key] = None
        elif any(isinstance(v, list) for v in values):
            # If any value is a list, flatten the structure
            flattened = []
            for v in values:
                if isinstance(v, list):
                    flattened.extend(v)
                else:
                    flattened.append(v)
            combined[key] = flattened
        else:
            combined[key] = values
    
    # Ensure standard result key exists for compatibility
    if "results" not in combined:
        combined["results"] = []
        
    return combined


class CompositeTransformation(BaseTransformation):
    """Transformation that composes multiple transformations.
    
    Applies multiple transformations in sequence, from right to left
    (mathematical function composition order). This enables complex
    transformation pipelines with a clean interface.
    """
    
    def __init__(self, *transforms: TransformProtocol) -> None:
        """Initialize with transforms to compose.
        
        Args:
            *transforms: Transformations to compose, applied from right to left
        """
        super().__init__(name="compose")
        self.transforms = transforms
        
    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Apply all transformations to a function.
        
        Args:
            fn: Function to transform
            
        Returns:
            Function with all transformations applied in composition order
        """
        result = fn
        # Apply transforms in reverse order (right to left)
        for transform in reversed(self.transforms):
            result = transform(result)
        # Preserve function identity for introspection
        return self._preserve_function_metadata(fn, result)


def compose(*transforms: TransformProtocol) -> TransformProtocol:
    """Compose multiple transformations into a single transformation.
    
    Args:
        *transforms: Transformations to compose
        
    Returns:
        Composite transformation that applies all transforms in sequence
        
    Example:
        ```python
        # Combine vmap and pmap
        vectorized_parallel = compose(vmap(batch_size=32), pmap(num_workers=4))
        
        # Apply to a function
        process_fn = vectorized_parallel(process_item)
        ```
    """
    return CompositeTransformation(*transforms)