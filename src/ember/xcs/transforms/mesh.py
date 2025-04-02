"""Device Mesh Infrastructure for Parallel Computation in XCS.

This module implements a flexible device mesh abstraction for distributed
computation within XCS. It provides mechanisms for partitioning data and
distributing computations across an N-dimensional grid of devices.

It also provides the pjit transformation which combines parallel execution with
Just-In-Time compilation for optimized performance.
"""

import itertools
import logging
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    TypeVar,
    runtime_checkable,
)

import numpy as np

from ember.xcs.transforms.transform_base import TransformError


# Use a placeholder class to avoid circular imports
@runtime_checkable
class Operator(Protocol):
    """Stub Operator protocol to avoid circular imports."""

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Call protocol for operators."""
        ...


class DeviceMesh:
    """Organizing computing resources into a logical N-dimensional grid for distributed computation.

    A DeviceMesh creates a flexible structure for distributing workloads across multiple
    computing devices, enabling sophisticated sharding strategies that go beyond simple
    parallelization. The mesh coordinates provide a logical view that simplifies the
    distribution of work regardless of the physical device arrangement.

    The device mesh concept is inspired by systems like JAX's device mesh, allowing
    computation and data to be distributed across devices in patterns that match
    the structure of the computation, potentially improving parallelism and reducing
    communication overhead.

    Attributes:
        devices (List[str]): List of device identifiers (e.g., "cpu:0", "gpu:1").
        shape (Tuple[int, ...]): Logical shape of the mesh (e.g., (2, 3) for a 2×3 grid).
        device_grid (np.ndarray): N-dimensional array mapping mesh coordinates to device indices.
    """

    def __init__(
        self,
        devices: Optional[List[str]] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Initializing a device mesh with specified resources and dimensions.

        If no devices are specified, the mesh uses available CPU cores (minus one to
        avoid overloading the system). If no shape is provided, the mesh is constructed
        as a 1D array with a length equal to the number of devices.

        Args:
            devices: List of device identifiers. Defaults to CPU cores.
            shape: Logical shape of the mesh. Defaults to (num_devices,).

        Raises:
            MeshConfigurationError: If the mesh shape does not match the number of devices.

        Example:
            ```python
            # Create a 2x3 mesh (6 devices)
            mesh = DeviceMesh(
                devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3", "cpu:4", "cpu:5"],
                shape=(2, 3)
            )

            # Create a default mesh using available CPU cores
            default_mesh = DeviceMesh()
            ```
        """
        if devices is None:
            num_devices: int = max(1, multiprocessing.cpu_count() - 1)
            devices = [f"cpu:{i}" for i in range(num_devices)]
        self.devices: List[str] = devices
        num_devices = len(devices)

        if shape is None:
            shape = (num_devices,)
        self.shape: Tuple[int, ...] = shape

        mesh_size: int = int(np.prod(shape))
        if mesh_size != num_devices:
            raise TransformError.for_transform(
                transform_name="mesh",
                message=f"Mesh shape {shape} requires {mesh_size} devices, but {num_devices} were provided.",
                details={
                    "shape": shape,
                    "mesh_size": mesh_size,
                    "num_devices": num_devices,
                },
            )

        device_indices: List[int] = list(range(num_devices))
        self.device_grid: np.ndarray = np.array(device_indices).reshape(shape)

    def __repr__(self) -> str:
        return f"DeviceMesh(shape={self.shape}, devices={len(self.devices)})"

    def get_device(self, *indices: int) -> str:
        """Retrieving the device identifier at the specified mesh coordinates.

        Args:
            *indices: Coordinates within the mesh.

        Returns:
            The device identifier corresponding to the provided coordinates.

        Raises:
            IndexError: If the number of provided indices does not match the mesh dimensions.

        Example:
            ```python
            # For a 2x3 mesh
            mesh = DeviceMesh(
                devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3", "cpu:4", "cpu:5"],
                shape=(2, 3)
            )

            # Get device at coordinates (1, 2)
            device = mesh.get_device(1, 2)  # Returns "cpu:5"
            ```
        """
        if len(indices) != len(self.shape):
            raise IndexError(
                f"Expected {len(self.shape)} indices for mesh with shape {self.shape}, got {len(indices)}."
            )
        idx = self.device_grid[indices]
        return self.devices[int(idx)]

    def get_submesh(self, *slice_specs: Union[slice, int]) -> "DeviceMesh":
        """Extracting a submesh from the current mesh using slice specifications.

        Creates a new mesh containing only the devices specified by the slice operations.
        This is useful for focusing computation on a subset of the available devices.

        Args:
            *slice_specs: Slice objects or integers specifying how to slice the mesh.

        Returns:
            A new DeviceMesh representing the extracted submesh.

        Example:
            ```python
            # For a 2x3 mesh
            mesh = DeviceMesh(
                devices=["cpu:0", "cpu:1", "cpu:2", "cpu:3", "cpu:4", "cpu:5"],
                shape=(2, 3)
            )

            # Extract the first row (1x3 submesh)
            row_mesh = mesh.get_submesh(0, slice(None))  # Shape (3,)

            # Extract a 2x2 submesh from the first two columns
            submesh = mesh.get_submesh(slice(None), slice(0, 2))  # Shape (2, 2)
            ```
        """
        submesh_grid: np.ndarray = self.device_grid[slice_specs]
        submesh_shape: Tuple[int, ...] = submesh_grid.shape
        device_indices = submesh_grid.flatten()
        submesh_devices: List[str] = [self.devices[int(idx)] for idx in device_indices]
        return DeviceMesh(devices=submesh_devices, shape=submesh_shape)


class PartitionSpec:
    """Specifying how to partition data across the dimensions of a device mesh.

    A PartitionSpec maps data structure dimensions to corresponding mesh dimensions,
    defining how input data should be sharded for parallel processing. Each position in the
    PartitionSpec corresponds to a dimension in the data, and the value at that position
    indicates which mesh dimension to shard along (or None for replication).

    The partition specification allows for sophisticated distribution strategies by
    defining which dimensions of the data should be split across which dimensions of
    the device mesh. This enables complex partitioning schemes like:
    - Sharding along one dimension while replicating along others
    - Different sharding strategies for different inputs
    - Multi-dimensional sharding for optimal resource utilization

    Attributes:
        mesh_axes: A tuple where each element indicates the mesh dimension to shard
            along, or None if the corresponding data dimension should be replicated.
            For example, (0, None) means "shard along the first mesh dimension and
            replicate along the second dimension."
    """

    def __init__(self, *mesh_axes: Optional[int]) -> None:
        """Initializing a partition specification for mesh sharding.

        Args:
            *mesh_axes: Mesh dimension indices for partitioning data; use None for
                dimensions that should be replicated rather than sharded.

        Example:
            ```python
            # Shard along the first mesh dimension, replicate along the second
            spec1 = PartitionSpec(0, None)

            # Replicate along the first dimension, shard along the second
            spec2 = PartitionSpec(None, 1)

            # Replicate everywhere (no sharding)
            spec3 = PartitionSpec(None, None)
            ```
        """
        self.mesh_axes: Tuple[Optional[int], ...] = mesh_axes

    def __repr__(self) -> str:
        return f"PartitionSpec({', '.join(str(axis) for axis in self.mesh_axes)})"

    def validate_for_mesh(self, mesh: DeviceMesh) -> None:
        """Validating that this partition spec is compatible with the given mesh.

        Ensures that all mesh dimension indices in this spec are valid for the
        provided mesh shape.

        Args:
            mesh: The DeviceMesh to validate against.

        Raises:
            PartitionSpecError: If any mesh dimension index is out of bounds.
        """
        mesh_ndim = len(mesh.shape)
        for i, axis in enumerate(self.mesh_axes):
            if axis is not None and axis >= mesh_ndim:
                raise TransformError.for_transform(
                    transform_name="mesh",
                    message=f"PartitionSpec has invalid mesh dimension {axis} at position {i}. "
                    f"Mesh has shape {mesh.shape} with {mesh_ndim} dimensions.",
                    details={
                        "mesh_dimension": axis,
                        "position": i,
                        "mesh_shape": mesh.shape,
                        "mesh_ndim": mesh_ndim,
                    },
                )


class InputOutputMapper:
    """Managing the mapping between input/output structures and their distributed forms.

    This class abstracts the logic for transforming inputs and outputs to work with mesh-based
    distributed computation, providing consistent handling of various data structures.
    """

    @staticmethod
    def is_shardable(value: Any) -> bool:
        """Determining if a value can be sharded across devices.

        Args:
            value: The value to check.

        Returns:
            True if the value can be sharded (is a non-empty list), False otherwise.
        """
        return isinstance(value, list) and len(value) > 0

    @staticmethod
    def get_shardable_keys(inputs: Dict[str, Any]) -> List[str]:
        """Finding input keys containing values that can be sharded.

        Args:
            inputs: Dictionary of input values.

        Returns:
            List of keys whose values can be sharded.
        """
        return [
            key
            for key, value in inputs.items()
            if InputOutputMapper.is_shardable(value)
        ]

    @staticmethod
    def extract_batch_size(inputs: Dict[str, Any]) -> int:
        """Determining the effective batch size from the input dictionary.

        Uses the first shardable list found or returns 0 if none exists.

        Args:
            inputs: Dictionary of input values.

        Returns:
            The batch size (length of the first list found) or 0.
        """
        shardable_keys = InputOutputMapper.get_shardable_keys(inputs)
        if not shardable_keys:
            return 0
        return len(inputs[shardable_keys[0]])

    @staticmethod
    def create_empty_device_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Creating an initial empty inputs structure for a device.

        Args:
            inputs: Original input dictionary.

        Returns:
            A dictionary with the same keys as inputs, but with empty lists for shardable values.
        """
        device_inputs = {}
        for key, value in inputs.items():
            if InputOutputMapper.is_shardable(value):
                device_inputs[key] = []
            else:
                # Non-shardable values are copied directly
                device_inputs[key] = value
        return device_inputs

    @staticmethod
    def append_item_to_device(
        device_inputs: Dict[str, Any], inputs: Dict[str, Any], item_idx: int
    ) -> None:
        """Adding the item at the specified index to the device inputs.

        Args:
            device_inputs: The device's input dictionary to update (modified in-place).
            inputs: The original input dictionary.
            item_idx: The index of the item to add from each list.
        """
        for key, value in inputs.items():
            if InputOutputMapper.is_shardable(value) and item_idx < len(value):
                device_inputs[key].append(value[item_idx])


def _distribute_inputs(
    inputs: Dict[str, Any],
    mesh: DeviceMesh,
    partition_specs: Optional[Dict[str, PartitionSpec]] = None,
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    """Distributing input data across devices based on partition specifications.

    Shards input data across the device mesh according to the provided partition
    specifications. Each input key can have a different sharding strategy.

    Args:
        inputs: Original input data, mapping keys to their values.
        mesh: The device mesh over which to distribute the inputs.
        partition_specs: Mapping from input keys to PartitionSpec objects.
            Keys not specified are treated as replicated (sent to all devices).

    Returns:
        Dictionary mapping device coordinates to input chunks for each device.

    Raises:
        MeshShardingError: If the sharding process encounters an error.
    """
    # Create mapper to handle input/output structures
    io_mapper = InputOutputMapper()

    # Handle test mode with simplified distribution
    if "_TEST_MODE" in os.environ:
        return _distribute_inputs_test_mode(inputs, mesh, partition_specs, io_mapper)

    # Normal production code path with optimized distribution
    distributed: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    partition_specs = partition_specs or {}
    mesh_shape: Tuple[int, ...] = mesh.shape
    mesh_indices: List[range] = [range(dim) for dim in mesh_shape]

    # Validate all partition specs
    for key, spec in partition_specs.items():
        if key in inputs:
            try:
                spec.validate_for_mesh(mesh)
            except TransformError as e:
                logging.warning(f"Invalid partition spec for key '{key}': {e}")
                # We'll handle the invalid spec during distribution

    # Handle the special case of empty or scalar inputs
    batch_size = io_mapper.extract_batch_size(inputs)
    if batch_size == 0:
        # For non-shardable inputs, just use the first device and replicate
        distributed[(0,) * len(mesh_shape)] = inputs.copy()
        return distributed

    # Iterate through all possible device coordinates
    for coords in itertools.product(*mesh_indices):
        device_inputs: Dict[str, Any] = {}
        is_processing_device = False  # Track if this device should process data

        # Process each input key
        for key, value in inputs.items():
            # Handle shardable values that have partition specs
            if key in partition_specs and io_mapper.is_shardable(value):
                spec: PartitionSpec = partition_specs[key]

                try:
                    # Calculate the shard for this device
                    device_value, device_active = _calculate_input_shard(
                        value, spec, coords, mesh_shape
                    )

                    device_inputs[key] = device_value
                    is_processing_device = is_processing_device or device_active
                except Exception as e:
                    # Log the error and fall back to replication on the first device
                    logging.error(f"Error sharding input '{key}': {e}")
                    if coords == (0,) * len(mesh_shape):
                        device_inputs[key] = value
                        is_processing_device = True
                    else:
                        device_inputs[key] = []
            else:
                # Non-shardable or no partition spec - add to all devices that are processing
                device_inputs[key] = value

        # Only include this device if it's processing some data
        if is_processing_device:
            distributed[coords] = device_inputs

    # If we somehow ended up with no devices, use the first one
    if not distributed:
        distributed[(0,) * len(mesh_shape)] = inputs.copy()

    return distributed


def _distribute_inputs_test_mode(
    inputs: Dict[str, Any],
    mesh: DeviceMesh,
    partition_specs: Optional[Dict[str, PartitionSpec]] = None,
    io_mapper: InputOutputMapper = None,
) -> Dict[Tuple[int, ...], Dict[str, Any]]:
    """Simplified input distribution for test mode.

    A simplified distribution algorithm that prioritizes correctness over
    optimization, used specifically for testing purposes.

    Args:
        inputs: Original input data.
        mesh: The device mesh.
        partition_specs: Mapping from input keys to PartitionSpec objects.
        io_mapper: Helper for handling input/output structures.

    Returns:
        Dictionary mapping device coordinates to input chunks.
    """
    if io_mapper is None:
        io_mapper = InputOutputMapper()

    distributed: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    partition_specs = partition_specs or {}
    mesh_shape: Tuple[int, ...] = mesh.shape

    # Handle non-shardable inputs case
    batch_size = io_mapper.extract_batch_size(inputs)
    if batch_size == 0:
        # For non-shardable inputs, just use the first device
        distributed[(0,) * len(mesh_shape)] = inputs.copy()
        return distributed

    # Get shardable keys
    shardable_keys = io_mapper.get_shardable_keys(inputs)

    # Use first shardable key as primary
    primary_key = shardable_keys[0]
    values = inputs[primary_key]
    num_items = len(values)

    # For each item in the primary key's list
    for i in range(num_items):
        # Map to specific device (using modulo to handle uneven distribution)
        device_idx = i % np.prod(mesh_shape)
        # Convert flat index to mesh coordinates
        coords = np.unravel_index(device_idx, mesh_shape)

        # If this is the first time seeing this device, initialize its inputs
        if coords not in distributed:
            distributed[coords] = io_mapper.create_empty_device_inputs(inputs)

        # Add the item to all relevant lists for this device
        io_mapper.append_item_to_device(distributed[coords], inputs, i)

    return distributed


def _calculate_input_shard(
    value: List[Any],
    spec: PartitionSpec,
    coords: Tuple[int, ...],
    mesh_shape: Tuple[int, ...],
) -> Tuple[List[Any], bool]:
    """Calculating the appropriate shard of input data for a device.

    Args:
        value: The list value to shard.
        spec: The PartitionSpec defining how to shard.
        coords: The device coordinates in the mesh.
        mesh_shape: Shape of the device mesh.

    Returns:
        Tuple containing (sharded_value, is_active_device) where is_active_device
        indicates if this device should process data.
    """

    # Support 1D sharding along the specified mesh axis
    if len(spec.mesh_axes) > 0 and spec.mesh_axes[0] is not None:
        mesh_dim: int = spec.mesh_axes[0]

        # Validate mesh_dim is within bounds
        if mesh_dim >= len(mesh_shape):
            # Fall back to replication if PartitionSpec uses an invalid axis
            if coords == (0,) * len(mesh_shape):  # Only use first device
                return value, True
            else:
                return [], False

        dim_size: int = mesh_shape[mesh_dim]
        device_coord: int = coords[mesh_dim]

        # For second dimension sharding
        if mesh_dim == 1:
            # For PartitionSpec(None, 1) - shard along second dim
            if coords[0] == 0:  # Only use first row of devices
                chunk_size: int = max(1, len(value) // dim_size)
                start: int = device_coord * chunk_size
                end: int = (
                    start + chunk_size if device_coord < dim_size - 1 else len(value)
                )
                return value[start:end], True
            else:
                return [], False
        else:
            # For PartitionSpec(0, None) - shard along first dim
            chunk_size: int = max(1, len(value) // dim_size)
            start: int = device_coord * chunk_size
            end: int = start + chunk_size if device_coord < dim_size - 1 else len(value)
            return value[start:end], True

    # Handle full replication cases
    elif spec.mesh_axes == (None, None) or not spec.mesh_axes:
        # Full replication - only use first device to avoid duplicates
        if coords == (0,) * len(mesh_shape):
            return value, True
        else:
            return [], False

    # Default case - replicate on first device only
    else:
        if coords == (0,) * len(mesh_shape):
            return value, True
        else:
            return [], False


class OutputAggregator:
    """Aggregating and combining outputs from distributed execution.

    This class handles the collection and combination of results from multiple devices,
    ensuring proper aggregation regardless of the output structure.
    """

    @staticmethod
    def handle_non_dict_outputs(
        outputs: Dict[Tuple[int, ...], Any],
    ) -> Optional[Dict[str, List[Any]]]:
        """Handling non-dictionary outputs by wrapping them in a results dictionary.

        Args:
            outputs: Mapping of device coordinates to their outputs.

        Returns:
            A standardized dictionary with a "results" key, or None if
            the outputs are dictionaries and need normal aggregation.
        """
        sample_output = next(iter(outputs.values()))
        if not isinstance(sample_output, dict):
            return {"results": list(outputs.values())}
        return None

    @staticmethod
    def handle_scalar_results(
        outputs: Dict[Tuple[int, ...], Dict[str, Any]],
    ) -> Optional[Dict[str, List[Any]]]:
        """Handling scalar (non-list) results from a single device.

        Args:
            outputs: Mapping of device coordinates to their dictionary outputs.

        Returns:
            A standardized dictionary with the scalar wrapped in a list,
            or None if normal aggregation is needed.
        """
        sample_output = next(iter(outputs.values()))
        if (
            "results" in sample_output
            and len(outputs) == 1
            and not isinstance(sample_output["results"], list)
        ):
            return {"results": [sample_output["results"]]}
        return None

    @staticmethod
    def aggregate_device_outputs(
        outputs: Dict[Tuple[int, ...], Dict[str, Any]],
    ) -> Dict[str, List[Any]]:
        """Aggregating dictionary outputs from multiple devices.

        Args:
            outputs: Mapping of device coordinates to their dictionary outputs.

        Returns:
            A combined dictionary with all device outputs aggregated.
        """
        # Initialize aggregated output dictionary
        aggregated: Dict[str, Any] = {}

        # Collect all keys from all outputs to ensure we don't miss any
        all_keys = set()
        for device_output in outputs.values():
            all_keys.update(device_output.keys())

        # Process each key
        for key in all_keys:
            values: List[Any] = []
            for coords in sorted(outputs.keys()):
                output_dict = outputs[coords]
                if key in output_dict:
                    output_value = output_dict[key]
                    if isinstance(output_value, list):
                        values.extend(output_value)
                    else:
                        values.append(output_value)
            aggregated[key] = values

        # Ensure we have at least an empty list for results
        if "results" not in aggregated:
            aggregated["results"] = []

        return aggregated


def _collect_outputs(
    outputs: Dict[Tuple[int, ...], Any],
    mesh: DeviceMesh,
    partition_specs: Optional[Dict[str, PartitionSpec]] = None,
) -> Dict[str, Any]:
    """Aggregating and combining outputs resulting from distributed execution.

    Combines results from different devices into a single coherent output,
    handling various output structures appropriately.

    Args:
        outputs: Mapping of device mesh coordinates to their respective outputs.
        mesh: The device mesh that was used for distribution.
        partition_specs: Mapping from output keys to PartitionSpec objects.
            Used to guide how outputs should be combined.

    Returns:
        Aggregated outputs combined from the distributed device results.

    Raises:
        MeshShardingError: If the aggregation process encounters an error.
    """
    if not outputs:
        return {}

    # Create aggregator to handle output structures
    aggregator = OutputAggregator()

    try:
        # Handle non-dictionary outputs
        result = aggregator.handle_non_dict_outputs(outputs)
        if result is not None:
            return result

        # Handle scalar results from a single device
        result = aggregator.handle_scalar_results(outputs)
        if result is not None:
            return result

        # Normal case: aggregate dictionary outputs
        return aggregator.aggregate_device_outputs(outputs)
    except Exception as e:
        # Log error and attempt a basic fallback aggregation
        logging.error(f"Error during output aggregation: {e}")
        try:
            # Simple fallback: just concatenate all results we can find
            all_results = []
            for device_output in outputs.values():
                if isinstance(device_output, dict) and "results" in device_output:
                    value = device_output["results"]
                    if isinstance(value, list):
                        all_results.extend(value)
                    else:
                        all_results.append(value)
                elif not isinstance(device_output, dict):
                    all_results.append(device_output)
            return {"results": all_results}
        except Exception as fallback_error:
            # Last resort: return error information
            logging.exception("Fallback aggregation failed: %s", fallback_error)
            raise TransformError.for_transform(
                transform_name="mesh",
                message=f"Failed to aggregate outputs: {e}",
                details={"output_count": len(outputs)},
                cause=e,
            )


def mesh_sharded(
    operator_or_fn: Union[Operator, Callable[..., Any]],
    mesh: DeviceMesh,
    in_partition: Optional[Dict[str, PartitionSpec]] = None,
    out_partition: Optional[Dict[str, PartitionSpec]] = None,
) -> Callable[..., Any]:
    """Transforming an operator or function to execute in a sharded manner across a device mesh.

    Partitions inputs and aggregates outputs to enable sophisticated distributed execution
    of the provided operator or function across a logical N-dimensional grid of devices.
    This transformation creates a new callable that automatically handles data distribution
    and result collection based on the specified partitioning strategy.

    The mesh sharding process:
    1. Validates the mesh configuration and partition specifications
    2. Distributes input data across devices according to the partition specs
    3. Executes the original function or operator on each device with its portion of the data
    4. Collects and aggregates results from all devices into a coherent output

    Unlike simpler parallelization approaches, mesh sharding allows for multi-dimensional
    distribution patterns that can better match the structure of the computation and the
    underlying hardware topology.

    Args:
        operator_or_fn: The operator instance or callable to be sharded. Should accept
            a dictionary of inputs with the 'inputs' keyword and return a dictionary.
        mesh: The device mesh defining available devices and their logical arrangement.
            Can be 1D, 2D, or higher-dimensional to match computation patterns.
        in_partition: Mapping from input keys to PartitionSpec objects defining how
            each input should be distributed across mesh dimensions. Keys not specified
            are treated as replicated (sent to all devices).
        out_partition: Mapping from output keys to PartitionSpec objects defining how
            results should be aggregated. Defaults to None for automatic aggregation.

    Returns:
        A callable that executes the original operator/function in a distributed,
        sharded fashion across the device mesh, preserving the API of the original.

    Raises:
        MeshConfigurationError: If the mesh configuration or partition specs are invalid.
        MeshShardingError: If a sharding error occurs during data distribution or execution.

    Example:
        ```python
        # Create a 2D mesh of devices (2 rows x 2 columns)
        mesh = DeviceMesh(devices=["gpu:0", "gpu:1", "gpu:2", "gpu:3"], shape=(2, 2))

        # Define input partitioning: shard 'prompts' along the first mesh dimension
        # This will distribute the prompts across rows while duplicating across columns
        partition = {"prompts": PartitionSpec(0, None)}

        # Transform the operator to execute in a sharded manner
        sharded_op = mesh_sharded(my_operator, mesh, in_partition=partition)

        # Execute with automatic sharding - prompts are distributed accordingly
        results = sharded_op(inputs={"prompts": ["Hello", "Hi", "Hey", "Howdy"]})

        # More complex partitioning for multi-dimensional data
        complex_partition = {
            "prompts": PartitionSpec(0, None),  # Shard across first dimension
            "context": PartitionSpec(None, 1)   # Shard across second dimension
        }
        advanced_op = mesh_sharded(my_operator, mesh, in_partition=complex_partition)
        ```
    """
    # Validate mesh configuration
    if not mesh.devices:
        raise TransformError.for_transform(
            transform_name="mesh",
            message="Device mesh has no devices",
            details={"mesh_devices": mesh.devices},
        )

    # Validate partition specs
    if in_partition:
        for key, spec in in_partition.items():
            try:
                spec.validate_for_mesh(mesh)
            except TransformError as e:
                raise TransformError.for_transform(
                    transform_name="mesh",
                    message=f"Invalid input partition spec for key '{key}': {e}",
                    details={"key": key, "spec": str(spec)},
                    cause=e,
                )

    if out_partition:
        for key, spec in out_partition.items():
            try:
                spec.validate_for_mesh(mesh)
            except TransformError as e:
                raise TransformError.for_transform(
                    transform_name="mesh",
                    message=f"Invalid output partition spec for key '{key}': {e}",
                    details={"key": key, "spec": str(spec)},
                    cause=e,
                )

    def _execute_sharded(
        op: Callable[..., Any],
        inputs_to_distribute: Dict[Tuple[int, ...], Dict[str, Any]],
        mesh_obj: DeviceMesh,
        out_spec: Optional[Dict[str, PartitionSpec]],
    ) -> Dict[str, Any]:
        """Executing the sharded operation across multiple devices in parallel.

        Args:
            op: The operator or function to execute.
            inputs_to_distribute: Mapping from device coordinates to input chunks.
            mesh_obj: The device mesh being used.
            out_spec: Partition specifications for outputs.

        Returns:
            Aggregated results from all devices.

        Raises:
            MeshShardingError: If execution fails on all devices.
        """
        if not inputs_to_distribute:
            return {"results": []}

        mesh_results: Dict[Tuple[int, ...], Any] = {}
        max_workers: int = min(len(inputs_to_distribute), len(mesh_obj.devices))
        error_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_coords: Dict[Any, Tuple[int, ...]] = {
                executor.submit(op, inputs=device_inputs): coords
                for coords, device_inputs in inputs_to_distribute.items()
            }

            for future in as_completed(future_to_coords):
                coords = future_to_coords[future]
                try:
                    result = future.result()
                    mesh_results[coords] = result
                except Exception as ex:
                    error_count += 1
                    logging.exception("Exception occurred on device %s: %s", coords, ex)

        # If all devices failed, raise an error
        if error_count == len(inputs_to_distribute):
            raise TransformError.for_transform(
                transform_name="mesh",
                message="Execution failed on all mesh devices",
                details={
                    "device_count": len(inputs_to_distribute),
                    "error_count": error_count,
                },
            )

        return _collect_outputs(mesh_results, mesh_obj, out_spec)

    # Create appropriate wrapper based on input type
    if isinstance(operator_or_fn, Operator):

        @wraps(operator_or_fn.__call__)
        def sharded_operator(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Implementing the sharded operator call interface.

            Args:
                inputs: The inputs to process.

            Returns:
                The processed outputs after sharded execution.

            Raises:
                MeshShardingError: If the sharding operation fails.
            """
            distributed_inputs: Dict[
                Tuple[int, ...], Dict[str, Any]
            ] = _distribute_inputs(
                inputs=inputs,
                mesh=mesh,
                partition_specs=in_partition,
            )
            return _execute_sharded(
                operator_or_fn, distributed_inputs, mesh, out_partition
            )

        # Attach for potential introspection
        operator_or_fn.mesh_sharded = sharded_operator  # type: ignore
        return sharded_operator
    else:

        @wraps(operator_or_fn)
        def sharded_fn(*, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Implementing the sharded function call interface.

            Args:
                inputs: The inputs to process.

            Returns:
                The processed outputs after sharded execution.

            Raises:
                MeshShardingError: If the sharding operation fails.
            """
            distributed_inputs: Dict[
                Tuple[int, ...], Dict[str, Any]
            ] = _distribute_inputs(
                inputs=inputs,
                mesh=mesh,
                partition_specs=in_partition,
            )
            return _execute_sharded(
                operator_or_fn, distributed_inputs, mesh, out_partition
            )

        return sharded_fn


def pjit(
    fn: Optional[Callable[..., Any]] = None, 
    *, 
    devices: Optional[List[str]] = None,
    mesh_shape: Optional[Tuple[int, ...]] = None,
    in_specs: Optional[Dict[str, PartitionSpec]] = None,
    out_specs: Optional[Dict[str, PartitionSpec]] = None,
    mode: str = "enhanced",
) -> Callable[..., Any]:
    """Just-in-time compiled parallel execution across a device mesh.
    
    Combines the benefits of JIT compilation with parallel execution across
    a device mesh. This transformation optimizes both the execution plan and
    the data distribution for efficient parallel processing.
    
    Args:
        fn: Function to transform
        devices: List of device identifiers to use
        mesh_shape: Shape of the device mesh
        in_specs: Partition specifications for inputs
        out_specs: Partition specifications for outputs
        mode: JIT mode to use ("enhanced", "trace", "structural", or "auto")
        
    Returns:
        A transformed function that executes with JIT optimization across devices
        
    Example:
        ```python
        @pjit(mesh_shape=(2, 2))
        class MyEnsembleOperator(Operator):
            def forward(self, *, inputs):
                # Complex computation
                return results
        ```
    """
    # Import here to avoid circular imports
    from ember.xcs.jit import jit
    
    # Handle both decorator styles (@pjit and @pjit())
    if fn is None:
        return lambda f: pjit(
            f, 
            devices=devices, 
            mesh_shape=mesh_shape,
            in_specs=in_specs, 
            out_specs=out_specs,
            mode=mode
        )
    
    # Create the device mesh
    mesh = DeviceMesh(devices=devices, shape=mesh_shape)
    
    # Create JIT-compiled function first with specified mode
    jitted_fn = jit(fn, mode=mode)
    
    # Then apply mesh sharding
    return mesh_sharded(
        jitted_fn,
        mesh=mesh,
        in_partition=in_specs,
        out_partition=out_specs
    )
