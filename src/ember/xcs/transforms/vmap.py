"""
Vectorized mapping transformation for XCS.

This module provides the vmap transformation, which applies an operator across
a batch dimension, similar to JAX's vmap but adapted for XCS's operator model
and execution environment.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from functools import wraps

from ember.core.registry.operator.base.operator_base import Operator


def _get_batch_size(inputs: Dict[str, Any], in_axes: Union[int, Dict[str, int]]) -> int:
    """
    Determine the batch size from the inputs and in_axes specification.
    
    Args:
        inputs: The batched inputs dictionary
        in_axes: Either a single integer specifying the batch axis for all inputs,
                or a dictionary mapping input names to their batch axes
    
    Returns:
        The determined batch size. Returns 1 for scalar inputs or empty lists.
    
    Raises:
        ValueError: If batch sizes are inconsistent
    """
    batch_sizes = []
    
    # First try to find explicit batch sizes from list inputs
    if isinstance(in_axes, int):
        # Use the same batch axis for all inputs.
        for _, value in inputs.items():
            if isinstance(value, list):
                batch_sizes.append(len(value))
    else:
        # Use different batch axes per input.
        for name, value in inputs.items():
            if name in in_axes and isinstance(value, list):
                batch_sizes.append(len(value))

    # If we have batch sizes, check for consistency
    if batch_sizes:
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"Inconsistent batch sizes detected: {batch_sizes}.")
        return batch_sizes[0]
    
    # If no batched dimensions were found, we treat as a single item
    # This handles both scalar inputs and empty lists
    return 1


def _prepare_batched_inputs(
    inputs: Dict[str, Any], 
    in_axes: Union[int, Dict[str, int]], 
    batch_size: int
) -> Dict[str, List[Any]]:
    """
    Organize inputs for batched processing.
    
    Args:
        inputs: The original inputs
        in_axes: Specification of batch axes
        batch_size: The determined batch size
    
    Returns:
        Dictionary with lists of unbatched inputs for each batch element
    """
    result = {}
    
    for key, value in inputs.items():
        # Skip non-batched inputs
        if isinstance(in_axes, dict) and key not in in_axes:
            # Replicate scalar values across batch
            result[key] = [value] * batch_size
            continue
            
        if not isinstance(value, list):
            # Replicate scalar values across batch
            result[key] = [value] * batch_size
            continue
            
        # For empty lists, create a list of empty lists
        if len(value) == 0 and batch_size > 0:
            result[key] = [[]] * batch_size
            continue
            
        # Handle batched inputs
        result[key] = value
    
    return result


def _combine_outputs(
    results: List[Any], 
    out_axes: Union[int, Dict[str, int]] = 0
) -> Dict[str, Any]:
    """
    Combine per-batch outputs into final batched result.
    
    Args:
        results: List of individual outputs from each batch element
        out_axes: Specification of output batch axes
    
    Returns:
        Batched outputs
    """
    # Handle empty results case
    if not results:
        return {}
    
    # If all results are dictionaries, combine them by key
    if all(isinstance(r, dict) for r in results):
        combined = {}
        keys = results[0].keys()
        
        for key in keys:
            # Extract values and flatten one level of nesting if needed
            values = []
            for result in results:
                if key in result:
                    value = result[key]
                    # If we have a list with a single item, extract it to avoid double nesting
                    if isinstance(value, list) and len(value) == 1:
                        values.append(value[0])
                    else:
                        values.append(value)
            
            # Always keep the original values for results to avoid string conversion
            combined[key] = values
        
        return combined
    
    # If results are not dictionaries, return as a list
    return {"result": results}


def vmap(
    operator_or_fn: Union[Operator, Callable], 
    in_axes: Union[int, Dict[str, int]] = 0, 
    out_axes: Union[int, Dict[str, int]] = 0
) -> Callable:
    """
    Vectorizing map for XCS operators.
    
    This transformation converts an operator that works on single elements into
    one that can work on batches of elements. The transformation handles each batch
    element independently and combines the results.
    
    Args:
        operator_or_fn: The Operator instance or function to vectorize
        in_axes: Specification of batch axes for inputs (default: 0)
        out_axes: Specification of batch axes for outputs (default: 0)
    
    Returns:
        A vectorized version of the operator or function
    
    Example:
        ```python
        # Create a vectorized operator that processes batches
        batched_op = vmap(my_operator)
        
        # Process a batch of inputs
        results = batched_op(inputs={"prompts": ["Hello", "Hi", "Hey"]})
        ```
    """
    def _execute_vectorized(fn, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Inner function that handles the vectorization logic."""
        batch_size = _get_batch_size(inputs, in_axes)
        
        # Handle empty inputs special cases
        empty_inputs = False
        if "prompts" not in inputs:
            empty_inputs = True
        elif isinstance(inputs.get("prompts"), list) and len(inputs.get("prompts")) == 0:
            empty_inputs = True
            
        if empty_inputs:
            # For empty inputs, return empty results list
            return {"results": []}
            
        batched_inputs = _prepare_batched_inputs(inputs, in_axes, batch_size)
        
        # Run operator on each batch item
        results = []
        for i in range(batch_size):
            batch_input = {k: v[i] for k, v in batched_inputs.items()}
            results.append(fn(inputs=batch_input))
        
        # Handle batched outputs
        return _combine_outputs(results, out_axes)
    
    if isinstance(operator_or_fn, Operator):
        # Create a wrapper if it's an operator
        @wraps(operator_or_fn.__call__)
        def vectorized_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return _execute_vectorized(fn=operator_or_fn, inputs=inputs)
            
        # Create a reference to the vectorized function
        operator_or_fn.vectorized = vectorized_operator
        return vectorized_operator
    else:
        # It's a function, wrap it directly
        @wraps(operator_or_fn)
        def vectorized_fn(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
            return _execute_vectorized(fn=operator_or_fn, inputs=inputs)
            
        return vectorized_fn