"""
Vectorized Mapping (vmap): Batch Processing Transformation

Implementing a generalized vectorization transformation for batch processing in the XCS
framework. This module enables batched computation by transforming functions
that process individual items into functions that process batches, similar to JAX's
vmap but adapted for Ember's Operator-based architecture.
"""

from __future__ import annotations

import functools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from ember.core.exceptions import TransformError

logger = logging.getLogger(__name__)

# Type variables for generic typing support
T = TypeVar("T")
U = TypeVar("U")
InputT = TypeVar("InputT", bound=Mapping[str, Any])
OutputT = TypeVar("OutputT")
ContainerT = TypeVar("ContainerT", bound=Sequence[Any])

# Type alias for axis specification
AxisSpec = Union[int, Dict[str, int], None]


def _get_batch_size(inputs: Mapping[str, Any], in_axes: AxisSpec) -> int:
    """Determining batch size from input structure and axis specification.

    Analyzing inputs to find batch dimensions based on the axis specification,
    checking for consistency across batched inputs to ensure proper vectorization.

    Args:
        inputs: Dictionary of input values, potentially containing batched data.
        in_axes: Specification of which inputs should be treated as batched.
            Can be an integer (applied to all inputs), a dictionary mapping
            specific keys to their batch axes, or None (no batching).

    Returns:
        The batch size derived from consistent batch dimensions.

    Raises:
        ValueError: If inconsistent batch sizes are detected across batched inputs.
    """
    batch_sizes = []

    # Case 1: Integer axis specification applies to all inputs
    if isinstance(in_axes, int):
        for key, value in inputs.items():
            if isinstance(value, (list, tuple)) and value:
                batch_sizes.append(len(value))

    # Case 2: Dictionary axis specification for specific inputs
    elif isinstance(in_axes, dict):
        for key, value in inputs.items():
            if key in in_axes and isinstance(value, (list, tuple)) and value:
                batch_sizes.append(len(value))

    # Case 3: No batch dimension specified, try to auto-detect
    elif in_axes is None:
        for key, value in inputs.items():
            if isinstance(value, (list, tuple)) and value:
                batch_sizes.append(len(value))

    # Check for consistent batch sizes
    if not batch_sizes:
        return 1  # No batch dimensions found, treat as single item

    unique_sizes = set(batch_sizes)
    if len(unique_sizes) > 1:
        raise TransformError.for_transform(
            transform_name="vmap",
            message=f"Inconsistent batch sizes detected across inputs: {sorted(list(unique_sizes))}. "
            f"All batched inputs must have the same length.",
            batch_sizes=sorted(list(unique_sizes)),
        )

    return batch_sizes[0]


def _prepare_batched_inputs(
    inputs: Mapping[str, Any], in_axes: AxisSpec, batch_size: int
) -> List[Dict[str, Any]]:
    """Restructuring inputs for element-wise batch processing.

    Transforming mixed batch and non-batch inputs into a format where each batch element
    has its own complete set of inputs, handling broadcasting of scalar values.

    Args:
        inputs: Original input dictionary with mixed batch and scalar values.
        in_axes: Specification of which inputs are batched and along which axis.
        batch_size: Expected size for all batched inputs.

    Returns:
        List of input dictionaries, one for each batch element.

    Raises:
        TransformError: If a batched input doesn't match the expected batch size.
    """
    # Prepare element-wise input dictionaries
    element_inputs: List[Dict[str, Any]] = [{} for _ in range(batch_size)]

    for key, value in inputs.items():
        is_batched = False

        # Determine if this input should be batched
        if isinstance(in_axes, int) and isinstance(value, (list, tuple)):
            is_batched = True
        elif (
            isinstance(in_axes, dict)
            and key in in_axes
            and isinstance(value, (list, tuple))
        ):
            is_batched = True
        elif in_axes is None and isinstance(value, (list, tuple)):
            is_batched = True

        # Handle batched input
        if is_batched and value:
            if len(value) != batch_size:
                raise TransformError.for_transform(
                    transform_name="vmap",
                    message=f"Input '{key}' has length {len(value)}, but batch size is {batch_size}.",
                    key=key,
                    actual_length=len(value),
                    batch_size=batch_size,
                )

            # Distribute batched values to each element's inputs
            for i in range(batch_size):
                element_inputs[i][key] = value[i]

        # Handle non-batched input (broadcast to all elements)
        else:
            for i in range(batch_size):
                element_inputs[i][key] = value

    return element_inputs


def _combine_outputs(results: Sequence[Any], out_axes: AxisSpec = 0) -> Dict[str, Any]:
    """Combining individual outputs into a cohesive batched result.

    Merging outputs from individual batch elements into a single result structure
    that preserves the semantics of the original function while supporting batching.

    Args:
        results: Sequence of individual outputs from batch element processing.
        out_axes: Configuration for how outputs should be combined along dimensions.

    Returns:
        Combined output dictionary with batched results.
    """
    if not results:
        return {}

    # Case 1: All results are dictionaries - combine by key
    if all(isinstance(r, Mapping) for r in results):
        # Get all unique keys across all result dictionaries
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())

        # Combine values for each key
        combined: Dict[str, Any] = {}
        for key in all_keys:
            # Collect values for this key from all results
            values = []
            for result in results:
                # Handle missing keys and unwrap single-item lists
                if key in result:
                    value = result[key]
                    if isinstance(value, list) and len(value) == 1:
                        values.append(value[0])
                    else:
                        values.append(value)

            # Store combined values
            combined[key] = values

        return combined

    # Case 2: Results are not all dictionaries - use standard result key
    return {"result": list(results)}


def vmap(
    fn: Callable[..., T], *, in_axes: AxisSpec = 0, out_axes: AxisSpec = 0
) -> Callable[..., Dict[str, Any]]:
    """Vectorizing a function across its inputs.

    Transforms a function that operates on single elements into one that efficiently
    processes multiple inputs in parallel. This transformation preserves the original
    function's semantics while automatically handling batch processing capabilities, similar
    to JAX's vmap but adapted for Ember's dictionary-based operators.

    The vectorization process automatically:
    1. Identifies batched and non-batched inputs based on the in_axes specification
    2. Computes a consistent batch size across all inputs
    3. Creates per-element input dictionaries with appropriate values
    4. Applies the original function to each element individually
    5. Combines the results into a properly structured batched output

    Args:
        fn: The function to vectorize. Should accept a dictionary of inputs with the
            'inputs' keyword and return a dictionary.
        in_axes: Specification of which inputs are batched and on which axis.
            If an integer, applies to all inputs. If a dict, specifies axes
            for specific keys. Keys not specified are treated as non-batch inputs
            and will be broadcast to all elements.
        out_axes: Configuration for how outputs should be combined along dimensions.
            Currently used primarily to ensure API consistency with other transforms.

    Returns:
        A vectorized version of the input function that handles batched inputs
        and produces batched outputs, preserving the original function's semantics.

    Raises:
        TransformError: If inconsistent batch sizes are detected across inputs.

    Example:
        ```python
        def process_item(*, inputs):
            return {"processed": transform(inputs["data"])}

        # Creating vectorized version
        batch_process = vmap(process_item)

        # Processing multiple items at once
        results = batch_process(inputs={"data": ["item1", "item2", "item3"]})
        # results == {"processed": ["transformed_item1", "transformed_item2", "transformed_item3"]}

        # Using a specific in_axes specification
        selective_batch = vmap(process_item, in_axes={"data": 0, "config": None})
        results = selective_batch(inputs={
            "data": ["item1", "item2", "item3"],
            "config": {"param": "value"}  # Will be broadcast to all elements
        })
        ```
    """

    @functools.wraps(fn)
    def vectorized_func(*, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """Vectorized version of the original function.

        Applying the original function to each batch element independently
        and combining the results into a batched output.

        Args:
            inputs: Batch inputs to process.

        Returns:
            Batched outputs with results for each input element.

        Raises:
            TransformError: If inconsistent batch sizes are detected.
        """
        # Handle empty input case
        if not inputs:
            return fn(inputs=inputs)  # Let the function handle empty inputs

        # Handle empty batched inputs case
        if any(isinstance(v, (list, tuple)) and not v for v in inputs.values()):
            # Delegate to the function to handle correctly
            return fn(inputs=inputs)

        # Determine batch size from inputs and axes specification
        batch_size = _get_batch_size(inputs=inputs, in_axes=in_axes)

        # Handle non-batched case (single item)
        if batch_size == 1:
            # For non-batched inputs, just call the original function
            return fn(inputs=inputs)

        # Prepare inputs for each batch element
        element_inputs = _prepare_batched_inputs(
            inputs=inputs, in_axes=in_axes, batch_size=batch_size
        )

        # Process each batch element
        results = []
        for batch_element in element_inputs:
            element_result = fn(inputs=batch_element)
            results.append(element_result)

        # Combine results from all batch elements
        return _combine_outputs(results=results, out_axes=out_axes)

    # Preserving metadata for introspection
    # Handle both functions and operator objects
    if hasattr(fn, "__name__"):
        vectorized_func.__name__ = f"vectorized_{fn.__name__}"
    else:
        # For operator objects, use class name
        vectorized_func.__name__ = f"vectorized_{fn.__class__.__name__}"

    if hasattr(fn, "__doc__") and fn.__doc__:
        vectorized_func.__doc__ = f"Vectorized version of: {fn.__doc__}"

    # Adding reference back to original function
    setattr(vectorized_func, "_original_func", fn)

    return vectorized_func
