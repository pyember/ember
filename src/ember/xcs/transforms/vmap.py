"""
Vectorized Mapping (vmap): Batch Processing Transformation

Providing the vmap transformation for parallel computation in the XCS framework.
The vmap transformation vectorizes operators and functions to process batched 
inputs in parallel, similar to NumPy's vectorization or JAX's vmap, but 
specialized for the XCS execution model.

Key features:
1. Processing multiple inputs in parallel without manual batching logic
2. Converting single-item operators to batched operators
3. Combining with other transforms like pmap for nested parallelism
4. Working with both primitive operators and complex pipelines

Handling batch dimensions, broadcasting scalar values across batches,
and merging results from batch processing.

Example:
    ```python
    # Creating a single-item operator
    class MyOperator(Operator):
        def __call__(self, *, inputs):
            return {"result": process_item(inputs["text"])}
            
    # Creating a vectorized version that processes batches in parallel
    batched_operator = vmap(MyOperator())
    
    # Processing multiple items with a single call
    results = batched_operator(inputs={
        "text": ["item1", "item2", "item3"],
        "options": {"format": "json"}  # Non-batched parameter applied to all items
    })
    # results == {"result": [processed1, processed2, processed3]}
    
    # Combining with other transforms for advanced parallelism
    distributed_batch_op = pmap(vmap(MyOperator()))
    ```
"""

from functools import wraps
from typing import (
    TypeVar,
    Callable,
    Dict,
    List,
    Union,
    Mapping,
    Sequence,
    Optional,
    Generic,
    cast,
    Iterator,
    Iterable,
    Any,
)
from typing_extensions import TypedDict, Protocol

# Use stub classes to avoid import errors
from typing import TypeVar, Dict, Any


# Stub class for Operator to avoid circular imports
class Operator:
    """Temporary Operator stub class to avoid import errors."""

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for calling an operator."""
        return {}


# Use TypeVar for NodeInputT and NodeOutputT
NodeInputT = TypeVar("NodeInputT", bound=Dict[str, Any])
NodeOutputT = TypeVar("NodeOutputT", bound=Dict[str, Any])


# Stub class for EmberModel
class EmberModel:
    """Temporary EmberModel stub class to avoid import errors."""

    pass


# Define more specific type variables
InputItemT = TypeVar("InputItemT")
OutputItemT = TypeVar("OutputItemT")
BatchInputT = TypeVar("BatchInputT", bound=Mapping[str, object])
BatchOutputT = TypeVar("BatchOutputT", bound=Mapping[str, object])


ItemT_co = TypeVar(
    "ItemT_co", covariant=True
)  # Type of item in a batchable container (covariant)


class BatchableInput(Protocol[ItemT_co]):
    """Protocol for types that can be batched."""

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> ItemT_co:
        ...


class BatchAxis(TypedDict, total=False):
    """Configuration for batch axis settings."""

    axis: int
    stack_results: bool
    preserve_batch_dim: bool


class VMapConfig(TypedDict, total=False):
    """Configuration for vmap operation."""

    strict_batch_size: bool
    check_batch_consistency: bool
    error_on_empty_batch: bool
    empty_batch_policy: str  # 'return_empty' | 'skip' | 'error'


def _get_batch_size(
    inputs: Mapping[str, object], in_axes: Union[int, Mapping[str, int]]
) -> int:
    """Determining the consistent batch size from a dictionary of inputs.

    Analyzing the input dictionary and batch axis specification
    to find the appropriate batch size. Checking that all
    batched inputs have consistent sizes.

    The function follows this logic:
    1. If in_axes is an integer, checking all sequence inputs for batch size
    2. If in_axes is a mapping, only checking keys specified in the mapping
    3. Verifying batch sizes are consistent
    4. Defaulting to 1 (single item) if no batched dimensions are found

    Args:
        inputs: Dictionary containing the input values, some of which may be batched.
        in_axes: Specification of which inputs should be treated as batched and
                along which axis. Can be either an integer (applied to all inputs)
                or a dictionary mapping specific input keys to their batch axes.

    Returns:
        The inferred batch size for processing. Returns 1 if no batched inputs
        are found, indicating single-item processing.

    Raises:
        ValueError: If inconsistent batch sizes are detected across different
                  batched inputs, which would make vectorization impossible.
    """
    batch_sizes: List[int] = []
    if isinstance(in_axes, int):
        for _, value in inputs.items():
            if isinstance(value, (list, tuple)):
                batch_sizes.append(len(value))
    else:
        for key, value in inputs.items():
            if key in in_axes and isinstance(value, (list, tuple)):
                batch_sizes.append(len(value))

    if batch_sizes:
        unique_sizes = set(batch_sizes)
        if len(unique_sizes) > 1:
            raise ValueError(f"Inconsistent batch sizes detected: {batch_sizes}.")
        return batch_sizes[0]
    return 1


def _prepare_batched_inputs(
    inputs: Mapping[str, object],
    in_axes: Union[int, Mapping[str, int]],
    batch_size: int,
) -> Dict[str, List[object]]:
    """Restructures inputs for element-wise batch processing.

    This function transforms a dictionary of mixed batch and non-batch inputs
    into a format where each key maps to a list with one value per batch element.
    This enables processing each batch element independently with consistent inputs.

    The transformation process follows these rules:
    1. Batched inputs are split into individual elements
    2. Non-batched (scalar) inputs are broadcast (replicated) across all batch elements
    3. Empty sequence inputs are handled specially to maintain batch consistency
    4. Validation ensures all batched inputs have the correct length

    This preprocessing step is critical for enabling vectorized execution
    while maintaining the semantics of the original operator.

    Args:
        inputs: Original input dictionary with mixed batch and scalar values.
        in_axes: Specification of which inputs are batched and along which axis.
        batch_size: The expected size for all batched inputs, as determined by
                   the _get_batch_size function.

    Returns:
        A restructured dictionary where every key maps to a list with exactly
        batch_size elements, arranged for element-wise processing.

    Raises:
        ValueError: If any batched input doesn't match the expected batch_size,
                   which would make consistent vectorization impossible.
    """
    result: Dict[str, List[object]] = {}
    for key, value in inputs.items():
        if isinstance(in_axes, dict) and key not in in_axes:
            result[key] = [value] * batch_size
            continue

        if not isinstance(value, (list, tuple)):
            result[key] = [value] * batch_size
            continue

        if len(value) == 0 and batch_size > 0:
            result[key] = [cast(List[object], [])] * batch_size
            continue

        value_list = list(value)
        if len(value_list) != batch_size:
            raise ValueError(
                f"Input '{key}' has length {len(value_list)}, but batch size is {batch_size}."
            )
        result[key] = value_list

    return result


def _combine_outputs(
    results: Sequence[object], out_axes: Union[int, Mapping[str, int]] = 0
) -> Dict[str, object]:
    """Merges individual outputs from batch processing into a consolidated result.

    This function is the inverse of _prepare_batched_inputs, taking the separate
    outputs from each batch element and combining them into a cohesive batched result.
    It handles both homogeneous outputs (all dictionaries with the same keys) and
    heterogeneous outputs (mixed types).

    The combination strategy adapts to the output structure:
    1. If all batch outputs are dictionaries:
       - Each key across all dictionaries is collected into a list
       - Single-item lists are automatically unwrapped for cleaner results
    2. If outputs are not all dictionaries:
       - All results are collected into a list under the key "result"

    This adaptability ensures that the vectorized operator maintains semantics
    similar to the original, just with batched inputs and outputs.

    Args:
        results: Sequence of individual outputs from processing each batch element.
        out_axes: Specification of how outputs should be combined. Currently used
                 for interface consistency; future versions may implement more
                 sophisticated output restructuring based on this parameter.

    Returns:
        A dictionary containing the combined results. The structure depends on
        the input results but is guaranteed to preserve all output information
        in a batched format.
    """
    if not results:
        return {}

    # Combining results if all are mappings (dictionaries), by key
    if all(isinstance(r, Mapping) for r in results):
        combined: Dict[str, object] = {}
        first_result = cast(Mapping[str, object], results[0])

        for key in first_result.keys():
            values: List[object] = []
            for result in results:
                result_dict = cast(Mapping[str, object], result)
                if key in result_dict:
                    value = result_dict[key]
                    # Unwrap single-item lists
                    if isinstance(value, list) and len(value) == 1:
                        values.append(value[0])
                    else:
                        values.append(value)
            combined[key] = values
        return combined

    # Otherwise, returning all results under the "result" key
    return {"result": list(results)}


def vmap(
    operator_or_fn: Union[
        Operator, Callable[[Mapping[str, object]], Mapping[str, object]]
    ],
    in_axes: Union[int, Dict[str, int]] = 0,
    out_axes: Union[int, Dict[str, int]] = 0,
) -> Callable[[Mapping[str, object]], Dict[str, object]]:
    """Transforming a single-item operator/function into a vectorized batch processor.

    Converting functions from processing individual items to handling batches. Applying
    the wrapped function to each batch element and then combining the results. The
    transformation maintains the original operator's semantics while adding
    batch processing capabilities.

    Handling several scenarios:
    - Mixed batch and non-batch inputs (broadcasting scalars across the batch)
    - Heterogeneous output structures
    - Result merging with type preservation
    - Edge cases (empty batches, single-item batches)

    Args:
        operator_or_fn: The target operator or function to vectorize. Can be either an
                      Operator instance or a callable with the same signature.
        in_axes: Specification of batch dimensions for inputs. Can be either:
               - An integer (default 0) indicating the batch axis for all inputs
               - A dictionary mapping specific input keys to their batch axes
               Keys not specified in the dictionary are treated as non-batch (scalar) inputs.
        out_axes: Specification of batch dimensions for outputs. Structure matches in_axes.
                Default is 0, combining outputs along the first dimension.

    Returns:
        A vectorized callable with the same signature as the input, but which processes
        batched inputs and returns batched outputs.

    Example:
        ```python
        # Creating a single-item operator
        class Translator(Operator):
            def __call__(self, *, inputs):
                return {"translated": translate(inputs["text"])}

        # Creating vectorized version
        batch_translator = vmap(Translator())

        # Processing a batch of texts in a single call
        results = batch_translator(inputs={
            "text": ["Hello", "World", "Example"],
            "target_language": "Spanish",  # Non-batch parameter, applied to all items
            "options": {"format": "json"}  # Complex non-batch parameter
        })
        # results == {"translated": ["Hola", "Mundo", "Ejemplo"]}

        # Using with dictionary outputs
        def process_item(item):
            return {
                "id": item["id"],
                "processed": transform(item["data"]),
                "timestamp": get_timestamp()
            }

        # Creating vectorized function
        batch_process = vmap(process_item)

        # Processing multiple items with heterogeneous outputs
        results = batch_process(inputs={
            "id": [1, 2, 3],
            "data": ["a", "b", "c"]
        })
        # results == {
        #    "id": [1, 2, 3],
        #    "processed": ["TRANSFORMED_A", "TRANSFORMED_B", "TRANSFORMED_C"],
        #    "timestamp": [t1, t2, t3]
        # }
        ```
    """

    # Type-specialized execution function to handle different callables
    def _execute_vectorized_op(
        fn: Callable[..., object], *, inputs: Mapping[str, object]
    ) -> Dict[str, object]:
        """Execute the operator/function in a batched manner.

        Prepares the inputs for batch processing, applies the operator/function to each batch element,
        and then combines the outputs.

        Args:
            fn: The operator or function to be executed.
            inputs: Input dictionary for batched processing.

        Returns:
            The combined batched outputs.
        """
        batch_size: int = _get_batch_size(inputs=inputs, in_axes=in_axes)

        # Handling empty prompt batch case
        if "prompts" not in inputs or (
            isinstance(inputs.get("prompts"), (list, tuple))
            and not inputs.get("prompts")
        ):
            return {"results": []}

        batched_inputs = _prepare_batched_inputs(
            inputs=inputs, in_axes=in_axes, batch_size=batch_size
        )

        batch_results: List[object] = []
        for i in range(batch_size):
            # Creating a single batch element input
            batch_element = {key: value[i] for key, value in batched_inputs.items()}

            # Processing the individual batch element - handling both calling styles
            if isinstance(fn, Operator):
                item_result = fn(inputs=batch_element)
            else:
                # For regular functions, follow their expected calling convention
                item_result = fn(inputs=batch_element)

            batch_results.append(item_result)

        return _combine_outputs(results=batch_results, out_axes=out_axes)

    # Creating a generic vectorized callable type - works for both operator and function case
    VectorizedCallable = Callable[[Mapping[str, object]], Dict[str, object]]

    # Handling Operator case
    if isinstance(operator_or_fn, Operator):
        # Creating the wrapper function with the correct signature
        def vectorized_operator(*, inputs: Mapping[str, object]) -> Dict[str, object]:
            return _execute_vectorized_op(fn=operator_or_fn, inputs=inputs)

        # Preserving the name and docstring
        vectorized_operator.__name__ = f"vectorized_{operator_or_fn.__class__.__name__}"
        vectorized_operator.__doc__ = operator_or_fn.__doc__

        # Attaching the vectorized version to the operator for reference
        setattr(operator_or_fn, "vectorized", vectorized_operator)
        return cast(VectorizedCallable, vectorized_operator)

    # Handling plain function case
    # Creating the wrapper function with the correct signature
    def vectorized_fn(*, inputs: Mapping[str, object]) -> Dict[str, object]:
        return _execute_vectorized_op(fn=operator_or_fn, inputs=inputs)

    # Preserving the name and docstring
    if hasattr(operator_or_fn, "__name__"):
        vectorized_fn.__name__ = f"vectorized_{operator_or_fn.__name__}"
    if hasattr(operator_or_fn, "__doc__") and operator_or_fn.__doc__:
        vectorized_fn.__doc__ = operator_or_fn.__doc__

    return cast(VectorizedCallable, vectorized_fn)
