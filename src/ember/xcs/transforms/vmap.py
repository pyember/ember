"""Vectorized mapping transformation with unified implementation.

Provides a unified implementation of the vmap transformation that vectorizes
functions to operate on batched inputs, following the standardized transformation
API and utilizing the common BaseTransformation foundation.
"""

import concurrent.futures
import functools
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TypeVar

from ember.xcs.transforms.transform_base import (
    BaseTransformation,
    BatchingOptions,
    TransformError,
    combine_outputs,
    get_batch_size,
    split_batch,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class VMapTransformation(BaseTransformation):
    """Vectorizing transformation for batched inputs.
    
    Transforms a function that operates on single elements into one
    that efficiently processes multiple inputs in parallel. The transformation
    preserves the original function's semantics while enabling batch processing.
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
        """Initialize the vectorizing transformation.
        
        Args:
            in_axes: Specification of which inputs are batched and on which axis.
                If an integer, applies to all inputs. If a dict, specifies axes
                for specific keys. Keys not specified are treated as non-batch inputs.
            out_axis: Axis in output for batched results
            batch_size: Optional maximum batch size for processing
            parallel: Whether to process batch elements in parallel
            max_workers: Maximum number of workers for parallel processing
        """
        super().__init__("vmap")
        
        self.options = BatchingOptions(
            in_axes=in_axes,
            out_axis=out_axis,
            batch_size=batch_size,
            parallel=parallel,
            max_workers=max_workers,
        )
        
        # Validate options
        self.options.validate()
    
    def __call__(self, fn: Callable[..., T]) -> Callable[..., Dict[str, Any]]:
        """Apply the vectorizing transformation to a function.
        
        Args:
            fn: Function to vectorize. Should accept a dictionary of inputs with the
                'inputs' keyword and return a dictionary.
                
        Returns:
            A vectorized version of the input function.
        """
        @functools.wraps(fn)
        def vectorized_fn(**kwargs: Any) -> Dict[str, Any]:
            """Vectorized version of the original function.
            
            Args:
                **kwargs: Keyword arguments including 'inputs'
                
            Returns:
                Dictionary with batched results
                
            Raises:
                TransformError: If inputs or execution is invalid
            """
            # Extract main inputs
            if "inputs" not in kwargs:
                raise TransformError.for_transform(
                    "vmap", 
                    "vmap requires an 'inputs' parameter"
                )
            
            inputs = kwargs["inputs"]
            if not isinstance(inputs, dict):
                raise TransformError.for_transform(
                    "vmap",
                    "vmap requires dict input"
                )
            
            # Get the batch size from inputs
            try:
                total_batch_size = get_batch_size(inputs, self.options.in_axes)
            except TransformError as e:
                raise TransformError.for_transform("vmap", str(e))
            
            # Apply size limit if specified
            batch_size = (
                min(total_batch_size, self.options.batch_size)
                if self.options.batch_size is not None
                else total_batch_size
            )
            
            # Test mode overrides parallel setting for deterministic behavior
            use_parallel = self.options.parallel and batch_size > 1
            if os.environ.get("_TEST_MODE") == "1":
                use_parallel = False
            
            # Process each batch element
            if use_parallel:
                results = self._process_parallel(
                    fn, inputs, kwargs, batch_size, self.options.max_workers
                )
            else:
                results = self._process_sequential(fn, inputs, kwargs, batch_size)
            
            # Combine results
            try:
                return combine_outputs(results, self.options.out_axis)
            except Exception as e:
                raise TransformError.for_transform(
                    "vmap",
                    f"Error combining outputs: {e}",
                    cause=e
                )
        
        # Add metadata for introspection
        return self._preserve_function_metadata(fn, vectorized_fn)
        
    def _process_sequential(
        self, 
        fn: Callable, 
        inputs: Dict[str, Any], 
        kwargs: Dict[str, Any],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Process batch elements sequentially.
        
        Args:
            fn: The function to apply
            inputs: Batched inputs
            kwargs: Additional keyword arguments
            batch_size: Effective batch size
            
        Returns:
            List of results for each batch element
        """
        results = []
        for i in range(batch_size):
            try:
                # Extract this batch element
                element_inputs = split_batch(inputs, self.options.in_axes, i)
                
                # Process with original function
                other_kwargs = {k: v for k, v in kwargs.items() if k != 'inputs'}
                result = fn(inputs=element_inputs, **other_kwargs)
                
                # Add to results
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing batch element {i}: {e}")
                # Re-raise with context
                raise TransformError.for_transform(
                    "vmap",
                    f"Error processing batch element {i}",
                    cause=e
                )
                
        return results
        
    def _process_parallel(
        self, 
        fn: Callable, 
        inputs: Dict[str, Any], 
        kwargs: Dict[str, Any],
        batch_size: int,
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Process batch elements with optimal execution engine.
        
        Args:
            fn: Function to execute
            inputs: Batch inputs
            kwargs: Additional keyword arguments
            batch_size: Size of the batch
            max_workers: Maximum concurrent operations
            
        Returns:
            Results from batch processing
        """
        # Import here to avoid circular dependencies
        from ember.xcs.utils.executor import Dispatcher
        
        # Extract executor type from environment
        executor_type = os.environ.get("XCS_EXECUTION_ENGINE", "auto")
        
        # Create dispatcher for parallel execution
        dispatcher = Dispatcher(
            max_workers=max_workers,
            timeout=None,
            fail_fast=True,
            executor=executor_type
        )
        
        try:
            # Prepare input dictionaries for each batch element
            input_dicts = []
            for i in range(batch_size):
                try:
                    # Extract this batch element
                    element_inputs = split_batch(inputs, self.options.in_axes, i)
                    
                    # Prepare input dict with kwargs
                    other_kwargs = {k: v for k, v in kwargs.items() if k != 'inputs'}
                    input_dict = {"inputs": element_inputs, **other_kwargs}
                    input_dicts.append(input_dict)
                except Exception as e:
                    logger.error(f"Error preparing batch element {i}: {e}")
                    raise TransformError.for_transform(
                        "vmap",
                        f"Error preparing batch element {i}",
                        cause=e
                    )
            
            # Execute across all inputs
            return dispatcher.map(fn, input_dicts)
        finally:
            dispatcher.close()


def vmap(
    fn: Optional[Callable[..., T]] = None, 
    *, 
    in_axes: Union[int, Dict[str, int]] = 0, 
    out_axis: int = 0,
    batch_size: Optional[int] = None,
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> Union[Callable[..., Dict[str, Any]], Callable[[Callable[..., T]], Callable[..., Dict[str, Any]]]]:
    """Vectorizing a function across its inputs.
    
    Transforms a function that operates on single elements into one
    that efficiently processes multiple inputs in parallel. The transformation
    preserves the original function's semantics while enabling batch processing.
    
    Args:
        fn: The function to vectorize. Should accept and return dictionaries.
        in_axes: Specification of which inputs are batched and on which axis.
            If an integer, applies to all inputs. If a dict, specifies axes
            for specific keys. Keys not specified are treated as non-batch inputs.
        out_axis: Axis in output for batched results
        batch_size: Optional maximum batch size for processing
        parallel: Whether to process batch elements in parallel
        max_workers: Maximum number of workers for parallel processing
    
    Returns:
        A vectorized version of the input function, or a decorator if fn is None.
    
    Example:
        ```python
        def process_item(*, inputs):
            return {"result": inputs["value"] * 2}
        
        # Vectorizing to process a batch
        batch_process = vmap(process_item)
        
        # Processing multiple items at once
        results = batch_process(inputs={"value": [1, 2, 3]})
        # results == {"result": [2, 4, 6]}
        ```
    """
    transformation = VMapTransformation(
        in_axes=in_axes,
        out_axis=out_axis,
        batch_size=batch_size,
        parallel=parallel,
        max_workers=max_workers,
    )
    
    if fn is None:
        return transformation
    return transformation(fn)