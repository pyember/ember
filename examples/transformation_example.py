"""
Example demonstrating the use of XCS transformations: vmap, pmap/pjit, and mesh.

This example shows how to use the various XCS transformation APIs to parallelize and
distribute computations across devices.
"""

import time
from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple

from ember.core.registry.operator.base.operator_base import Operator
# Note: Unused import removed to adhere to clean code practices.
from ember.xcs.transforms import (
    vmap,
    pmap,
    pjit,
    DeviceMesh,
    PartitionSpec,
    mesh_sharded,
)


def _time_function_call(
    callable_obj: Callable[..., Any], **kwargs: Any
) -> Tuple[float, Any]:
    """Execute a callable with keyword arguments and measure its execution time.

    Args:
        callable_obj: The function (or callable) to execute.
        **kwargs: Arbitrary keyword arguments to pass to the callable.

    Returns:
        A tuple containing:
            - The elapsed time in seconds.
            - The result returned by the callable.
    """
    start: float = perf_counter()
    result: Any = callable_obj(**kwargs)
    elapsed: float = perf_counter() - start
    return elapsed, result


class SimpleOperator(Operator):
    """A simple operator that processes input prompts by appending a suffix.

    This operator simulates a processing delay and returns the processed results.
    """

    def __call__(self, *, inputs: Dict[str, Any]) -> Dict[str, List[str]]:
        """Process the provided inputs by concatenating a processing suffix.

        Args:
            inputs: A dictionary with input data. The key 'prompts' should map to a
                single string or a list of strings to be processed.

        Returns:
            A dictionary with the key 'results' mapping to a list of processed prompts.
        """
        prompts: Any = inputs.get("prompts", [])
        # Simulate processing delay.
        time.sleep(0.1)
        if isinstance(prompts, list):
            processed_results: List[str] = [f"{prompt} -> processed" for prompt in prompts]
        else:
            processed_results = [f"{prompts} -> processed"]
        return {"results": processed_results}


def demonstrate_vmap() -> None:
    """Demonstrate the vmap (vectorized mapping) transformation.

    This function creates a vectorized version of a simple operator and compares
    its performance against sequential processing by timing both approaches.
    """
    print("\n=== VMAP Demonstration ===")

    simple_operator: SimpleOperator = SimpleOperator()
    vectorized_operator: Callable[..., Any] = vmap(simple_operator)

    batch_inputs: Dict[str, List[str]] = {
        "prompts": [
            "Hello, World!",
            "Vectorized mapping",
            "Processing a batch",
            "XCS transformations",
        ]
    }

    # Time sequential processing: apply the operator separately for each prompt.
    start_seq: float = perf_counter()
    sequential_results: List[Dict[str, Any]] = [
        simple_operator(inputs={"prompts": prompt}) for prompt in batch_inputs["prompts"]
    ]
    sequential_time: float = perf_counter() - start_seq
    print(f"Sequential processing time: {sequential_time:.4f}s")

    # Time vectorized processing: apply the operator once across all inputs.
    start_vec: float = perf_counter()
    vectorized_results: Dict[str, Any] = vectorized_operator(inputs=batch_inputs)
    vectorized_time: float = perf_counter() - start_vec
    print(f"Vectorized processing time: {vectorized_time:.4f}s")

    # Display results from the vectorized operator.
    print("\nResults from vectorized operator:")
    for result in vectorized_results.get("results", []):
        print(f"  {result}")


def demonstrate_pmap() -> None:
    """Demonstrate the pmap (parallel mapping) transformation.

    This function creates a parallelized operator and compares its performance on a batch
    of inputs against the sequential execution of the operator.
    """
    print("\n=== PMAP Demonstration ===")

    simple_operator: SimpleOperator = SimpleOperator()
    parallel_operator: Callable[..., Any] = pmap(simple_operator)

    batch_inputs: Dict[str, List[str]] = {
        "prompts": [
            "Hello, World!",
            "Parallel mapping",
            "Processing concurrently",
            "XCS transformations",
        ]
    }

    # Time sequential processing on the batch.
    sequential_time, sequential_results = _time_function_call(
        simple_operator, inputs=batch_inputs
    )
    print(f"Sequential processing time: {sequential_time:.4f}s")

    # Time parallelized processing on the batch.
    parallel_time, parallel_results = _time_function_call(
        parallel_operator, inputs=batch_inputs
    )
    print(f"Parallel processing time: {parallel_time:.4f}s")
    if parallel_time > 0:
        speedup: float = sequential_time / parallel_time
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Parallel processing time too small to calculate speedup.")

    # Display results from the parallel operator.
    print("\nResults from parallel operator:")
    for result in parallel_results.get("results", []):
        print(f"  {result}")


def demonstrate_mesh() -> None:
    """Demonstrate the device mesh transformation.

    This function creates a 2D device mesh and a corresponding mesh-sharded operator to
    partition and process inputs across simulated devices.
    """
    print("\n=== Device Mesh Demonstration ===")

    simple_operator: SimpleOperator = SimpleOperator()
    device_mesh: DeviceMesh = DeviceMesh(shape=(2, 2))
    print(f"Created mesh: {device_mesh}")

    partition_spec: Dict[str, PartitionSpec] = {
        "prompts": PartitionSpec(0, None)  # Shard along the first mesh dimension.
    }

    sharded_operator: Callable[..., Any] = mesh_sharded(
        simple_operator, device_mesh, in_partition=partition_spec
    )

    batch_inputs: Dict[str, List[str]] = {
        "prompts": [
            "Hello, World!",
            "Mesh sharding",
            "Processing across devices",
            "XCS transformations",
            "More prompts",
            "For demonstration",
            "Of mesh sharding",
            "Capabilities",
        ]
    }

    # Time sequential processing on the batch.
    sequential_time, sequential_results = _time_function_call(
        simple_operator, inputs=batch_inputs
    )
    print(f"Sequential processing time: {sequential_time:.4f}s")

    # Time mesh-sharded processing on the batch.
    sharded_time, sharded_results = _time_function_call(
        sharded_operator, inputs=batch_inputs
    )
    print(f"Mesh-sharded processing time: {sharded_time:.4f}s")
    if sharded_time > 0:
        speedup: float = sequential_time / sharded_time
        print(f"Speedup: {speedup:.2f}x")
    else:
        print("Mesh-sharded processing time too small to calculate speedup.")

    # Display results from the mesh-sharded operator.
    print("\nResults from mesh-sharded operator:")
    for result in sharded_results.get("results", []):
        print(f"  {result}")


def main() -> None:
    """Run all transformation demonstrations."""
    print("XCS Transformation Examples")

    demonstrate_vmap()
    demonstrate_pmap()
    demonstrate_mesh()

    print("\nAll transformations demonstrated successfully!")


if __name__ == "__main__":
    main()