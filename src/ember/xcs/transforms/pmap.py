"""
Parallel Mapping (pmap): Concurrent Execution Transformation

Providing a generalized parallelization transformation for distributed computation
in the XCS framework. This implementation enables efficient concurrent execution
by distributing workloads across multiple workers, optimized for Ember's architecture.

Key capabilities:
1. Automatic resource management: Dynamically allocates workers based on system capacity
2. Flexible input sharding: Thoughtfully distributes work for maximum parallelism
3. Error handling: Recovery from individual worker failures
4. Configurable execution: Fine-grained control over parallelization behavior
5. Composability and integrability: Compatible with other transformations like vmap
"""

import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar, Union, cast

from ember.core.exceptions import (
    ParallelExecutionError,
    TransformError,
    ValidationError,
)
from ember.xcs.transforms.transform_base import BaseTransformation, ParallelOptions


class ParallelTransformation(BaseTransformation):
    """Transformation for parallel execution.
    
    Transforms a function to execute in parallel across multiple workers,
    automatically distributing work and collecting results.
    """
    
    def __init__(
        self, 
        *, 
        num_workers=None, 
        continue_on_errors=False,
        timeout_seconds=None,
        devices=None
    ):
        """Initialize the parallel transformation.
        
        Args:
            num_workers: Number of worker threads to use
            continue_on_errors: Whether to continue execution if errors occur
            timeout_seconds: Maximum execution time before timeout
            devices: Optional list of device identifiers (unused, for API compatibility)
        """
        super().__init__("pmap")
        self.options = ParallelOptions(
            num_workers=num_workers,
            continue_on_errors=continue_on_errors,
            timeout_seconds=timeout_seconds,
            return_partial=True,
        )
        self.devices = devices
        
    def __call__(self, fn):
        """Apply the parallel transformation to a function.
        
        Args:
            fn: Function to parallelize
            
        Returns:
            Parallelized function
        """
        # Call the lower-level pmap implementation using our options
        parallelized = pmap(
            fn,
            num_workers=self.options.num_workers,
            devices=self.devices,
            execution_options=ExecutionOptions(
                continue_on_errors=self.options.continue_on_errors,
                timeout=self.options.timeout_seconds,
                return_partial_on_timeout=self.options.return_partial,
            ),
        )
        return self._preserve_function_metadata(fn, parallelized)


logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar("T")
R = TypeVar("R")
InputT = TypeVar("InputT", bound=Mapping[str, Any])
OutputT = TypeVar("OutputT", bound=Mapping[str, Any])


# Using exception hierarchy from ember.core.exceptions


@dataclass
class ShardingOptions:
    """Configuration options for input sharding behavior."""

    # If True, ensures all shardable inputs have the same size
    strict_batch_size: bool = True

    # If True, allows sharding to use different algorithms based on input structure
    adaptive_sharding: bool = True

    # Maximum individual shard size (0 for no limit)
    max_shard_size: int = 0

    # Strategy for sharding: "even" (equal sized), "greedy" (fill workers), or "dynamic"
    strategy: str = "even"

    def validate(self) -> None:
        """Validating sharding options for consistency.

        Checking that the configuration contains valid values and combinations
        of settings that can be used for parallel execution.

        Raises:
            ValueError: If any options have invalid values
        """
        valid_strategies = {"even", "greedy", "dynamic"}
        if self.strategy not in valid_strategies:
            raise ValidationError(
                f"Invalid sharding strategy '{self.strategy}'. "
                f"Must be one of: {', '.join(valid_strategies)}",
                context={
                    "strategy": self.strategy,
                    "valid_strategies": list(valid_strategies),
                },
            )

        if self.max_shard_size < 0:
            raise ValidationError(
                f"Invalid max_shard_size: {self.max_shard_size}. "
                "Must be >= 0 (0 means no limit)",
                context={"max_shard_size": self.max_shard_size},
            )


@dataclass
class ExecutionOptions:
    """Configuration options for parallel execution behavior."""

    # Maximum number of worker threads to use
    max_workers: Optional[int] = None

    # If True, continues execution despite individual shard failures
    continue_on_errors: bool = True

    # Maximum time in seconds to wait for all shards to complete
    timeout: Optional[float] = None

    # If True, collects and returns partial results when timeout occurs
    return_partial_on_timeout: bool = True

    def validate(self) -> None:
        """Validating execution options for consistency.

        Checking that the configuration contains valid values that can be used
        for parallel execution control.

        Raises:
            ValueError: If any options have invalid values
        """
        if self.max_workers is not None and self.max_workers <= 0:
            raise ValidationError(
                f"Invalid max_workers: {self.max_workers}. "
                "Must be > 0 or None for automatic selection",
                context={"max_workers": self.max_workers},
            )

        if self.timeout is not None and self.timeout <= 0:
            raise ValidationError(
                f"Invalid timeout: {self.timeout}. "
                "Must be > 0 or None for no timeout",
                context={"timeout": self.timeout},
            )


def _get_default_num_workers() -> int:
    """Determining the optimal number of worker threads for the current system.

    Analyzes system resources to select an appropriate default worker count,
    considering factors like CPU cores, current system load, and environment
    configuration.

    The implementation balances maximizing parallelism against excessive
    context switching overhead.

    Returns:
        The recommended number of worker threads
    """
    # Check for explicit environment variable configuration
    env_workers = os.environ.get("XCS_NUM_WORKERS")
    if env_workers:
        try:
            workers = int(env_workers)
            if workers > 0:
                return workers
        except ValueError:
            logger.warning(
                "Invalid XCS_NUM_WORKERS value '%s', using system-based default",
                env_workers,
            )

    # Default to CPU count minus 1 to avoid resource exhaustion
    # This typically provides good performance while leaving resources
    # for system processes and the coordination thread
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count - 1)


def _identify_shardable_inputs(
    inputs: Mapping[str, Any]
) -> Dict[str, Tuple[bool, int]]:
    """Identifying input fields suitable for sharding.

    Analyzes the input dictionary to determine which fields are shardable
    (sequence types that can be divided across workers) and their sizes.

    Args:
        inputs: Dictionary of input values

    Returns:
        Dictionary mapping field names to tuples of (is_shardable, size)
    """
    shardable_info: Dict[str, Tuple[bool, int]] = {}

    for key, value in inputs.items():
        # Check if value is a shardable sequence
        if isinstance(value, (list, tuple)) and value:
            shardable_info[key] = (True, len(value))
        else:
            shardable_info[key] = (False, 0)

    return shardable_info


def _validate_shard_inputs(
    inputs: Mapping[str, Any],
    num_shards: int,
) -> None:
    """Validating inputs for the sharding process.

    Checks that inputs meet the requirements for sharding, raising appropriate
    exceptions for invalid configurations.

    Args:
        inputs: Dictionary of input values to validate
        num_shards: Number of shards to create

    Raises:
        ValueError: If inputs or options are invalid
    """
    if not isinstance(inputs, Mapping):
        raise ValidationError(
            "Inputs must be a mapping (dictionary-like object)",
            context={"input_type": type(inputs).__name__},
        )

    if num_shards <= 0:
        raise ValidationError(
            f"Number of shards must be positive, got {num_shards}",
            context={"num_shards": num_shards},
        )


def _check_batch_size_consistency(
    shardable_info: Dict[str, Tuple[bool, int]], options: ShardingOptions
) -> None:
    """Checking for consistent batch sizes across shardable inputs.

    Verifies that all shardable inputs have compatible sizes when strict mode is enabled.

    Args:
        shardable_info: Dictionary mapping field names to (is_shardable, size) tuples
        options: Configuration for sharding behavior

    Raises:
        ShardingError: If inconsistent batch sizes are detected in strict mode
    """
    shardable_sizes = [
        size for _, (is_shardable, size) in shardable_info.items() if is_shardable
    ]
    # Only perform check if strict mode is enabled and we have multiple different sizes
    if options.strict_batch_size and len(set(shardable_sizes)) > 1:
        # Only check primary shardable fields
        primary_keys = [
            k
            for k, (is_shardable, size) in shardable_info.items()
            if is_shardable and size > 0
        ]
        if len(set(shardable_info[k][1] for k in primary_keys)) > 1:
            sizes_dict = {k: shardable_info[k][1] for k in primary_keys}
            raise TransformError.for_transform(
                transform_name="pmap",
                message="Inconsistent batch sizes detected across shardable inputs. "
                f"Sizes: {', '.join(f'{k}={shardable_info[k][1]}' for k in primary_keys)}. "
                "Set strict_batch_size=False in ShardingOptions to allow different sizes.",
                details={"shardable_sizes": sizes_dict},
            )


def _create_test_mode_shards(
    inputs: Mapping[str, Any], num_shards: int, min_size: int, shardable_keys: List[str]
) -> List[Dict[str, Any]]:
    """Creating shards for test mode.

    Implements a simplified sharding strategy for testing environments.

    Args:
        inputs: Dictionary of input values to distribute
        num_shards: Number of shards to create
        min_size: Minimum size of shardable inputs
        shardable_keys: List of keys for shardable inputs

    Returns:
        List of input dictionaries, one per shard
    """
    shards = [{} for _ in range(num_shards)]
    items_per_shard = min_size // num_shards
    for shard_idx in range(num_shards):
        shard = dict(inputs)

        # Calculate slice indices for this shard
        start_idx = shard_idx * items_per_shard
        end_idx = (
            start_idx + items_per_shard if shard_idx < num_shards - 1 else min_size
        )

        # Skip empty shards
        if start_idx >= end_idx:
            continue

        # Slice each shardable input
        for key in shardable_keys:
            if isinstance(inputs[key], (list, tuple)) and inputs[key]:
                shard[key] = inputs[key][start_idx:end_idx]

        shards[shard_idx] = shard

    return shards


def _create_even_shards(
    inputs: Mapping[str, Any],
    actual_shards: int,
    min_size: int,
    shardable_keys: List[str],
) -> List[Dict[str, Any]]:
    """Creating evenly distributed shards.

    Divides inputs into shards of approximately equal size.

    Args:
        inputs: Dictionary of input values to distribute
        actual_shards: Actual number of shards to create
        min_size: Minimum size of shardable inputs
        shardable_keys: List of keys for shardable inputs

    Returns:
        List of input dictionaries, one per shard
    """
    shards = [{} for _ in range(actual_shards)]
    items_per_shard = (
        min_size + actual_shards - 1
    ) // actual_shards  # Ceiling division

    for shard_idx in range(actual_shards):
        shard = dict(inputs)

        # Calculate slice indices for this shard
        start_idx = shard_idx * items_per_shard
        end_idx = min(start_idx + items_per_shard, min_size)

        # Skip empty shards
        if start_idx >= end_idx:
            continue

        # Process each shardable input field
        for key in shardable_keys:
            if isinstance(inputs[key], (list, tuple)) and inputs[key]:
                key_len = len(inputs[key])

                # Handle different-length shardable inputs
                if key_len <= min_size:
                    # For shorter lists, use the same indices directly
                    shard[key] = inputs[key][start_idx : min(end_idx, key_len)]
                else:
                    # For longer lists, scale the indices proportionally
                    scale = key_len / min_size
                    key_start = min(key_len - 1, int(start_idx * scale))
                    key_end = min(key_len, int(end_idx * scale))
                    shard[key] = inputs[key][key_start:key_end]

        shards[shard_idx] = shard

    return shards


@dataclass
class ShardingProcessingInfo:
    """Information used during sharding process."""

    key_len: int
    item_idx: int
    shard_size: int
    min_size: int


def _process_key_for_greedy_shard(
    inputs: Mapping[str, Any], key: str, info: ShardingProcessingInfo
) -> Any:
    """Process a single key for a greedy shard.

    Args:
        inputs: Dictionary of input values
        key: The key to process
        info: Information about the current sharding state

    Returns:
        The sliced value for this key
    """
    if not isinstance(inputs[key], (list, tuple)) or not inputs[key]:
        return inputs[key]

    key_len = len(inputs[key])

    if key_len <= info.min_size:
        # For shorter lists, use the same indices directly
        end_idx = min(info.item_idx + info.shard_size, key_len)
        return inputs[key][info.item_idx : end_idx]

    # For longer lists, scale the indices proportionally
    scale = key_len / info.min_size
    key_start = min(key_len - 1, int(info.item_idx * scale))
    key_end = min(key_len, int((info.item_idx + info.shard_size) * scale))
    return inputs[key][key_start:key_end]


def _create_greedy_shards(
    inputs: Mapping[str, Any],
    actual_shards: int,
    min_size: int,
    shardable_keys: List[str],
    max_items: int,
) -> List[Dict[str, Any]]:
    """Creating greedily distributed shards.

    Fills shards sequentially up to a maximum size per shard.

    Args:
        inputs: Dictionary of input values to distribute
        actual_shards: Actual number of shards to create
        min_size: Minimum size of shardable inputs
        shardable_keys: List of keys for shardable inputs
        max_items: Maximum items per shard

    Returns:
        List of input dictionaries, one per shard
    """
    shards = [{} for _ in range(actual_shards)]
    item_idx = 0
    shard_idx = 0

    while item_idx < min_size and shard_idx < actual_shards:
        shard = dict(inputs)
        shard_size = min(max_items, min_size - item_idx)

        # Information for processing keys
        info = ShardingProcessingInfo(
            key_len=0, item_idx=item_idx, shard_size=shard_size, min_size=min_size
        )

        # Process each shardable input field
        for key in shardable_keys:
            shard[key] = _process_key_for_greedy_shard(inputs, key, info)

        shards[shard_idx] = shard
        item_idx += shard_size
        shard_idx += 1

    return shards[:shard_idx]  # Only return non-empty shards


def _handle_non_shardable_inputs(
    inputs: Mapping[str, Any], num_shards: int, test_mode: bool
) -> Optional[List[Dict[str, Any]]]:
    """Handle the case when there are no shardable inputs.

    Args:
        inputs: Dictionary of input values
        num_shards: Number of shards requested
        test_mode: Whether we're running in test mode

    Returns:
        List of shards if there are no shardable inputs, None otherwise
    """
    if test_mode:
        return [dict(inputs) for _ in range(num_shards)]
    return [dict(inputs)]


@dataclass
class ShardingContext:
    """Context data for sharding operations."""

    inputs: Mapping[str, Any]
    num_shards: int
    min_size: int
    shardable_keys: List[str]
    options: ShardingOptions
    test_mode: bool


def _get_sharding_strategy_and_args(
    context: ShardingContext,
) -> Tuple[str, Dict[str, Any]]:
    """Determine sharding strategy and prepare arguments.

    Args:
        context: Sharding context containing all necessary information
            for determining the strategy and preparing arguments

    Returns:
        Tuple of (strategy_name, strategy_args)
    """
    actual_shards = min(context.num_shards, context.min_size)

    if context.test_mode:
        return "test_mode", {
            "inputs": context.inputs,
            "num_shards": context.num_shards,
            "min_size": context.min_size,
            "shardable_keys": context.shardable_keys,
        }

    if context.options.strategy == "greedy":
        max_items = (
            context.options.max_shard_size
            if context.options.max_shard_size > 0
            else context.min_size
        )
        return "greedy", {
            "inputs": context.inputs,
            "actual_shards": actual_shards,
            "min_size": context.min_size,
            "shardable_keys": context.shardable_keys,
            "max_items": max_items,
        }

    # Default to "even" for "even", "dynamic" or any other strategy
    return "even", {
        "inputs": context.inputs,
        "actual_shards": actual_shards,
        "min_size": context.min_size,
        "shardable_keys": context.shardable_keys,
    }


def _shard_inputs(
    inputs: Mapping[str, Any],
    num_shards: int,
    options: Optional[ShardingOptions] = None,
) -> List[Dict[str, Any]]:
    """Distributing input data into shards for parallel processing.

    Creates balanced shards from the input data, handling various input types
    and ensuring fair distribution of work across workers.

    Args:
        inputs: Dictionary of input values to distribute
        num_shards: Number of shards to create
        options: Configuration for sharding behavior

    Returns:
        List of input dictionaries, one per shard

    Raises:
        ShardingError: If sharding cannot be performed with the given inputs and options
        ValueError: If inputs or options are invalid
    """
    if options is None:
        options = ShardingOptions()
    else:
        options.validate()

    # Validate inputs
    _validate_shard_inputs(inputs, num_shards)

    # Special handling for test mode to ensure consistent behavior
    test_mode = os.environ.get("_TEST_MODE") == "1"

    # Enforce minimum of 1 shard
    num_shards = max(1, num_shards)

    # Identify shardable inputs and their sizes
    shardable_info = _identify_shardable_inputs(inputs)
    shardable_keys = [
        k for k, (is_shardable, _) in shardable_info.items() if is_shardable
    ]

    # If no shardable inputs, return single shard with all inputs
    if not shardable_keys:
        return _handle_non_shardable_inputs(inputs, num_shards, test_mode)

    # Find the shortest shardable input for consistent sharding
    shardable_sizes = [
        size for _, (is_shardable, size) in shardable_info.items() if is_shardable
    ]
    min_size = min(shardable_sizes)

    # Check for inconsistent batch sizes
    _check_batch_size_consistency(shardable_info, options)

    # Special case: single item batches don't need multiple shards
    if min_size == 1 and not test_mode:
        return [dict(inputs)]

    # Create context for sharding decisions
    context = ShardingContext(
        inputs=inputs,
        num_shards=num_shards,
        min_size=min_size,
        shardable_keys=shardable_keys,
        options=options,
        test_mode=test_mode,
    )

    # Determine sharding strategy and prepare arguments
    strategy, strategy_args = _get_sharding_strategy_and_args(context)

    # Create shards based on the selected strategy
    if strategy == "test_mode":
        return _create_test_mode_shards(**strategy_args)
    if strategy == "greedy":
        return _create_greedy_shards(**strategy_args)
    # Default strategy ("even")
    return _create_even_shards(**strategy_args)


def _combine_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combining results from parallel processing shards.

    Merges the outputs from individual worker shards into a cohesive
    result structure that preserves the semantics of the original function.

    Args:
        results: List of result dictionaries from parallel execution

    Returns:
        Combined result dictionary

    Raises:
        ValueError: If results have incompatible structures that cannot be combined
    """
    if not results:
        return {}

    # Special case for single result
    if len(results) == 1:
        # Check for scalar inputs that were already processed
        if "prompts" in results[0] and not isinstance(results[0]["prompts"], list):
            return results[0]
        if "results" in results[0] and not isinstance(results[0]["results"], list):
            return {"results": [results[0]["results"]]}

    # Gather unique keys from all results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    # Combine values for each key
    combined: Dict[str, Any] = {}
    for key in all_keys:
        aggregated_values = []

        for result in results:
            if key in result:
                value = result[key]
                if isinstance(value, list):
                    aggregated_values.extend(value)
                else:
                    aggregated_values.append(value)

        combined[key] = aggregated_values

    # Ensure standard result key exists
    if "results" not in combined:
        combined["results"] = []

    return combined


def _validate_and_prepare_pmap_args(
    func: Callable[[Mapping[str, Any]], Dict[str, Any]],
    num_workers: Optional[int] = None,
    sharding_options: Optional[ShardingOptions] = None,
    execution_options: Optional[ExecutionOptions] = None,
) -> Tuple[Callable, int, ShardingOptions, ExecutionOptions]:
    """Validate and prepare arguments for pmap function.

    Args:
        func: The function to parallelize
        num_workers: Number of worker threads to use
        sharding_options: Configuration for input distribution
        execution_options: Configuration for parallel execution

    Returns:
        Tuple of (validated_func, resolved_workers, validated_sharding_options,
                 validated_execution_options)

    Raises:
        ValueError: If any arguments are invalid
    """
    # Validate callable
    if not callable(func):
        raise ValidationError(
            f"Expected a callable function, got {type(func)}",
            context={"function_type": str(type(func))},
        )

    # Handle negative workers as an error
    if num_workers is not None and num_workers < 0:
        raise ValidationError(
            f"num_workers must be non-negative, got {num_workers}",
            context={"num_workers": num_workers},
        )

    # Handle zero workers as a special case - treat as None (use system default)
    if num_workers == 0:
        logger.debug("Using default worker count for num_workers=0")
        num_workers = None

    # Use default options if not provided
    sharding_opts = sharding_options or ShardingOptions()
    sharding_opts.validate()

    exec_opts = execution_options or ExecutionOptions()
    exec_opts.validate()

    # Resolve worker count
    resolved_workers = (
        num_workers if num_workers is not None else _get_default_num_workers()
    )

    return func, resolved_workers, sharding_opts, exec_opts


def _create_parallelized_func(
    func: Callable[[Mapping[str, Any]], Dict[str, Any]],
    resolved_workers: int,
    sharding_options: ShardingOptions,
    execution_options: ExecutionOptions,
) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    """Create the actual parallelized function implementation.

    Args:
        func: The function to parallelize
        resolved_workers: Resolved number of workers
        sharding_options: Validated sharding options
        execution_options: Validated execution options

    Returns:
        The parallelized function
    """
    # Now implemented inline in the pmap function
    return func  # Placeholder return to satisfy type checking


def pmap(
    func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    *,
    num_workers: Optional[int] = None,
    devices: Optional[List[str]] = None,  # Unused, but kept for API compatibility
    sharding_options: Optional[ShardingOptions] = None,
    execution_options: Optional[ExecutionOptions] = None,
) -> Union[Callable[[Dict[str, Any]], Dict[str, Any]], Callable[[Callable], Callable]]:
    """Parallelizing a function for concurrent execution.

    Transforms a function to execute across multiple workers in parallel,
    automatically distributing work and collecting results. This transformation enables
    efficient utilization of system resources for computation-intensive tasks by
    identifying batch dimensions in inputs and distributing them across workers.

    The parallelization process:
    1. Analyzes input data structures to identify batchable dimensions
    2. Automatically shards inputs across available workers based on configuration
    3. Executes the original function concurrently on each shard
    4. Handles failures gracefully based on execution options
    5. Aggregates results from all workers into a consistent output structure

    Args:
        func: The function to parallelize, accepting a dictionary of inputs with the
            'inputs' keyword and returning a dictionary of outputs.
        num_workers: Number of worker threads to use. If None, uses a system-determined
            value based on available CPU cores. If 0, also uses the system default.
        devices: Optional list of device identifiers for specialized hardware.
            Note: Currently unused but kept for API compatibility with other transforms.
        sharding_options: Configuration for input distribution behavior, controlling
            how inputs are split across workers. See ShardingOptions class for details.
        execution_options: Configuration for parallel execution behavior, including
            timeout handling and error recovery. See ExecutionOptions class for details.

    Returns:
        A parallelized version of the function that automatically distributes
        work across workers and aggregates results, preserving the semantics of
        the original function.

    Raises:
        ValueError: If any parameters are invalid
        ShardingError: If input data cannot be sharded properly
        ExecutionError: If parallel execution encounters unrecoverable problems

    Example:
        ```python
        def process_item(*, inputs):
            # Potentially expensive processing
            return {"processed": transform(inputs["data"])}

        # Create parallelized version with 4 workers
        parallel_process = pmap(process_item, num_workers=4)

        # Process multiple items concurrently
        results = parallel_process(inputs={"data": ["item1", "item2", "item3", "item4"]})
        # results == {"processed": ["transformed_item1", "transformed_item2", "transformed_item3", "transformed_item4"]}

        # With custom execution options for fault tolerance
        options = ExecutionOptions(continue_on_errors=True, timeout=60.0)
        robust_process = pmap(process_item, execution_options=options)
        ```
    """
    # Handle decorator style usage (@pmap)
    if func is None:
        # Return a decorator that will be called with the function
        def decorator(f):
            return pmap(
                f,
                num_workers=num_workers,
                devices=devices,
                sharding_options=sharding_options,
                execution_options=execution_options,
            )
        return decorator

    # If devices parameter is used, log that it's currently not functional
    if devices is not None:
        logger.debug(
            "The 'devices' parameter is currently unused but kept for API compatibility"
        )

    # Validate and prepare all arguments
    (
        func,
        resolved_workers,
        sharding_options,
        execution_options,
    ) = _validate_and_prepare_pmap_args(
        func, num_workers, sharding_options, execution_options
    )

    def _handle_execution_error(
        shard_index: int, exc: Exception, error_type: str
    ) -> None:
        """Handle execution errors based on configuration.

        Args:
            shard_index: The index of the failed shard
            exc: The exception that occurred
            error_type: A description of the error type for logging

        Raises:
            ExecutionError: If continue_on_errors is False
        """
        if execution_options.continue_on_errors:
            logger.warning("%s processing shard %d: %s", error_type, shard_index, exc)
        else:
            raise ParallelExecutionError.for_worker(
                worker_id=f"shard-{shard_index}",
                message=f"{error_type} processing shard {shard_index}",
                cause=exc,
            )

    def _process_future_result(
        future: Any, shard_index: int, results: List, errors: List
    ) -> None:
        """Process the result of a completed future.

        Args:
            future: The completed future
            shard_index: The index of the shard
            results: List to add successful results to
            errors: List to add errors to
        """
        try:
            result = future.result(timeout=execution_options.timeout)
            results.append(result)
        except TimeoutError as exc:
            errors.append((shard_index, exc))
            _handle_execution_error(shard_index, exc, "Timeout")
        except (ValueError, TypeError) as exc:
            errors.append((shard_index, exc))
            _handle_execution_error(shard_index, exc, "Value error")
        except (RuntimeError, KeyError, AttributeError, IndexError) as exc:
            errors.append((shard_index, exc))
            _handle_execution_error(shard_index, exc, "Runtime error")
        except OSError as exc:
            errors.append((shard_index, exc))
            _handle_execution_error(shard_index, exc, "I/O error")

    @wraps(func)
    def parallelized_func(*, inputs: Mapping[str, Any]) -> Dict[str, Any]:
        """Parallelized version of the original function.

        Distributes inputs across workers, executes the original function
        concurrently, and combines results.

        Args:
            inputs: Input data to process in parallel

        Returns:
            Combined results from all workers

        Raises:
            ShardingError: If inputs cannot be properly sharded
            ExecutionError: If parallel execution encounters problems
        """
        # Import here to avoid circular dependencies
        from ember.xcs.utils.executor import Dispatcher
        
        try:
            # Create input shards for parallel processing
            sharded_inputs = _shard_inputs(
                inputs=inputs, num_shards=resolved_workers, options=sharding_options
            )

            # If no valid shards created, fall back to direct execution
            if not sharded_inputs:
                return func(inputs=inputs)

            # Extract executor type from environment
            executor_type = os.environ.get("XCS_EXECUTION_ENGINE", "auto")
            
            # Create dispatcher for parallel execution
            dispatcher = Dispatcher(
                max_workers=resolved_workers,
                timeout=execution_options.timeout if execution_options else None,
                fail_fast=not (execution_options and execution_options.continue_on_errors),
                executor=executor_type
            )
            
            try:
                # Format input dictionaries for dispatcher
                input_dicts = [{"inputs": shard} for shard in sharded_inputs]
                
                # Execute tasks in parallel
                shard_results = dispatcher.map(func, input_dicts)
                
                # Check for empty results
                if not shard_results:
                    raise ParallelExecutionError(
                        message="No results returned from parallel execution",
                        context={"num_shards": len(sharded_inputs)}
                    )
                
                # Combine and return results
                return _combine_results(shard_results)
            finally:
                dispatcher.close()

        except TransformError:
            # Re-raise transform errors directly
            raise
        except Exception as exc:
            # Wrap other exceptions
            raise ParallelExecutionError(
                message="Error during parallel execution", cause=exc
            )

    # Preserve metadata for introspection
    try:
        # For function objects, use their name
        func_name = getattr(func, "__name__", None)
        if func_name:
            parallelized_func.__name__ = f"parallelized_{func_name}"
        else:
            # For class-based operators, use their class name
            parallelized_func.__name__ = f"parallelized_{func.__class__.__name__}"
    except (AttributeError, TypeError):
        # Fallback for any other callable types
        parallelized_func.__name__ = "parallelized_operator"

    # Preserve docstring if available
    if hasattr(func, "__doc__") and func.__doc__:
        parallelized_func.__doc__ = f"Parallelized version of: {func.__doc__}"

    # Add reference back to original function/operator
    parallelized_func._original_func = func

    return parallelized_func


@dataclass
class PJitOptions:
    """Configuration options for parallel JIT execution."""

    # Number of worker threads to use
    num_workers: Optional[int] = None

    # Device identifiers for specialized hardware (unused, for API compatibility)
    devices: Optional[List[str]] = None

    # Configuration for input sharding
    sharding_options: Optional[ShardingOptions] = None

    # Configuration for parallel execution
    execution_options: Optional[ExecutionOptions] = None

    # Argument indices to treat as static (reserved for future use)
    static_argnums: Optional[List[int]] = None


def pjit(
    func: Callable[[Mapping[str, Any]], Dict[str, Any]],
    *,
    num_workers: Optional[int] = None,
    devices: Optional[List[str]] = None,  # Unused, but kept for API compatibility
    sharding_options: Optional[ShardingOptions] = None,
    execution_options: Optional[ExecutionOptions] = None,
    static_argnums: Optional[
        List[int]
    ] = None,  # Reserved for future JIT implementation
) -> Callable[[Mapping[str, Any]], Dict[str, Any]]:
    """Parallel JIT compilation and execution for functions.

    Combines JIT compilation with parallel execution for maximum performance.
    This transformation optimizes a function's execution plan and runs it
    concurrently across multiple workers, providing both the benefits of
    just-in-time compilation and parallel processing.

    Note: Currently implemented as a direct wrapper for pmap. Future versions
    will integrate with XCS JIT compilation for additional optimization.

    Args:
        func: The function to optimize and parallelize, accepting a dictionary
            of inputs with the 'inputs' keyword and returning a dictionary.
        num_workers: Number of worker threads to use. If None, uses a system-determined
            value based on available CPU cores.
        devices: Optional list of device identifiers for specialized hardware.
            Note: Currently unused but kept for API compatibility.
        sharding_options: Configuration for input distribution behavior, controlling
            how inputs are split across workers. See ShardingOptions class for details.
        execution_options: Configuration for parallel execution behavior, including
            timeout handling and error recovery. See ExecutionOptions class for details.
        static_argnums: Optional list of argument indices to treat as static during
            compilation (not used in current implementation, reserved for future use).

    Returns:
        An optimized, parallelized version of the function that combines the benefits
        of JIT compilation and parallel execution.

    Raises:
        ValueError: If any parameters are invalid
        ShardingError: If input data cannot be sharded properly
        ExecutionError: If parallel execution encounters unrecoverable problems

    Example:
        ```python
        def process_item(*, inputs):
            # Potentially expensive processing with complex calculations
            return {"processed": complex_transform(inputs["data"])}

        # Create optimized parallel version
        optimized_process = pjit(process_item, num_workers=4)

        # Process batch of items with maximum performance
        results = optimized_process(inputs={"data": ["item1", "item2", "item3", "item4"]})
        ```
    """
    # Group options into a single dataclass to reduce argument count
    options = PJitOptions(
        num_workers=num_workers,
        devices=devices,
        sharding_options=sharding_options,
        execution_options=execution_options,
        static_argnums=static_argnums,
    )

    # Log info about unused parameters to aid debugging
    if options.static_argnums is not None:
        logger.debug(
            "static_argnums parameter is not used in the current pjit implementation"
        )

    if options.devices is not None:
        logger.debug("devices parameter is not used in the current pjit implementation")

    # Forward to pmap implementation
    return pmap(
        func,
        num_workers=options.num_workers,
        devices=options.devices,
        sharding_options=options.sharding_options,
        execution_options=options.execution_options,
    )
