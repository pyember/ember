"""Transforms Integration Example.

This example demonstrates how to effectively combine Ember XCS transforms
(vmap, pmap, structural_jit) with execution options to create optimized
processing pipelines. It shows different integration patterns and
provides performance comparisons.

The example illustrates:
1. Proper operator definition following Ember conventions
2. Applying transforms in effective combinations
3. Controlling execution with execution_options
4. Performance patterns and best practices

For a comprehensive explanation of the transforms and execution options, see:
- docs/xcs/TRANSFORMS.md
- docs/xcs/EXECUTION_OPTIONS.md
- docs/xcs/JIT_OVERVIEW.md

To run:
    uv run python src/ember/examples/xcs/transforms_integration_example.py
"""

import logging
import random
import time
from typing import Any, ClassVar, Dict

from ember.core.registry.operator.base.operator_base import Operator

# Import Ember APIs
from ember.core.registry.specification.specification import Specification
from ember.xcs.engine import execution_options
from ember.xcs.jit import jit
from ember.xcs.transforms import pmap, vmap

# Use the structural JIT strategy directly
structural_jit = jit.structural

###############################################################################
# Basic Component Operators
###############################################################################


class TextProcessor(Operator[Dict[str, Any], Dict[str, Any]]):
    """Processes text with configurable operations.

    This operator demonstrates proper field declarations and type annotations.
    """

    # Class-level specification
    specification: ClassVar[Specification] = Specification()

    # Field declarations
    process_type: str
    max_length: int

    def __init__(
        self, *, process_type: str = "tokenize", max_length: int = 100
    ) -> None:
        """Initialize the text processor.

        Args:
            process_type: The type of processing to perform
            max_length: Maximum length of text to process
        """
        self.process_type = process_type
        self.max_length = max_length

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process text according to configured options.

        This operator works on a single text item, not a batch. When used with
        transforms like vmap or pmap, those transforms are responsible for
        properly batching or distributing the inputs and combining the outputs.

        Args:
            inputs: Dictionary containing the text to process

        Returns:
            Dictionary with processed text and metadata
        """
        # Get input text with validation - expect a string, not a collection
        text = inputs.get("text", "")

        # Validate input type - provide clear errors for misuse
        if not isinstance(text, str):
            # This should never happen with properly configured transforms
            # But we provide a clear error message if it does
            raise TypeError(
                f"TextProcessor expects text input to be a string, got {type(text).__name__}. "
                "This may indicate incorrect transform configuration."
            )

        # Simulate different processing types with varying costs
        if self.process_type == "tokenize":
            # Fast operation
            time.sleep(0.01)
            tokens = text[: self.max_length].split() if text else []
            processed = f"Tokens: {tokens}"
        elif self.process_type == "analyze":
            # Medium cost operation
            time.sleep(0.05)
            processed = f"Analysis: {text[:self.max_length]}"
        elif self.process_type == "summarize":
            # Expensive operation
            time.sleep(0.1)
            summary_length = min(20, len(text)) if text else 0
            processed = f"Summary: {text[:summary_length]}..."
        else:
            processed = f"Unknown operation: {text[:self.max_length]}"

        return {
            "processed": processed,
            "original_length": len(text),
            "process_type": self.process_type,
        }


class FeatureExtractor(Operator[Dict[str, Any], Dict[str, Any]]):
    """Extracts features from processed text.

    This operator works as a secondary stage in a processing pipeline.
    """

    # Class-level specification
    specification: ClassVar[Specification] = Specification()

    # Field declarations
    feature_type: str
    num_features: int

    def __init__(self, *, feature_type: str = "basic", num_features: int = 5) -> None:
        """Initialize the feature extractor.

        Args:
            feature_type: Type of features to extract
            num_features: Maximum number of features to extract
        """
        self.feature_type = feature_type
        self.num_features = num_features

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from processed text.

        Args:
            inputs: Dictionary containing processed text

        Returns:
            Dictionary with extracted features
        """
        # Get processed text from previous stage
        processed = inputs.get("processed", "")
        process_type = inputs.get("process_type", "unknown")

        # Simulate feature extraction based on feature_type
        time.sleep(0.03)  # Base cost

        # Generate simulated features
        features = []
        for i in range(min(self.num_features, 10)):
            # Using hash of combination to ensure deterministic but varied output
            feature_hash = hash(f"{processed}_{i}_{self.feature_type}")
            random.seed(feature_hash)
            feature_val = random.random()
            features.append((f"feature_{i}", feature_val))

        return {
            "features": features,
            "feature_type": self.feature_type,
            "source_type": process_type,
            "feature_count": len(features),
        }


###############################################################################
# Pipeline Definition (for structural_jit example)
###############################################################################


@structural_jit
class TextAnalysisPipeline(Operator[Dict[str, Any], Dict[str, Any]]):
    """Two-stage text analysis pipeline with JIT optimization.

    This composite operator demonstrates proper structure for @structural_jit,
    with properly declared fields for sub-operators.
    """

    # Class-level specification
    specification: ClassVar[Specification] = Specification()

    # Field declarations for sub-operators
    processor: TextProcessor
    extractor: FeatureExtractor

    def __init__(
        self, *, process_type: str = "tokenize", feature_type: str = "basic"
    ) -> None:
        """Initialize the pipeline with configured sub-operators.

        Args:
            process_type: Type of text processing to perform
            feature_type: Type of features to extract
        """
        # Create sub-operators
        self.processor = TextProcessor(process_type=process_type, max_length=150)
        self.extractor = FeatureExtractor(feature_type=feature_type, num_features=8)

    def forward(self, *, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full pipeline on inputs.

        The @structural_jit decorator will optimize this execution flow
        by analyzing the operator structure and dependencies.

        Args:
            inputs: Dictionary containing the text to process

        Returns:
            Dictionary with processed text and extracted features
        """
        # Two-stage pipeline that will be optimized by structural_jit
        processed = self.processor(inputs=inputs)
        features = self.extractor(inputs=processed)

        # Combine results (optional, just to show final combination)
        result = {
            "text_length": processed.get("original_length", 0),
            "process_type": processed.get("process_type", "unknown"),
            "features": features.get("features", []),
            "feature_type": features.get("feature_type", "unknown"),
            "feature_count": features.get("feature_count", 0),
        }

        return result


###############################################################################
# Transform Integration Examples
###############################################################################


def demonstrate_basic_transforms():
    """Demonstrate basic transforms applied individually."""
    print("\n=== BASIC TRANSFORM EXAMPLES ===")

    # Create a simple text processor
    processor = TextProcessor(process_type="tokenize", max_length=100)

    # Sample input
    sample_text = "This is a sample text for demonstrating transforms in Ember XCS."

    # 1. No transforms (baseline)
    start_time = time.time()
    result = processor(inputs={"text": sample_text})
    baseline_time = time.time() - start_time
    print(f"Baseline (no transforms): {baseline_time:.4f}s")
    print(f"Result: {result['processed']}")

    # 2. With vmap (batch processing)
    print("\n--- Vectorized Mapping (vmap) Example ---")
    vectorized_processor = vmap(processor)
    batch_texts = [
        "First sample text for batch processing.",
        "Second sample with different content.",
        "Third sample to demonstrate batching capabilities.",
    ]

    # Explain vmap input/output structure
    print("vmap transforms a single-item operator to process multiple inputs at once.")
    print(f"Input: {len(batch_texts)} text items to be processed in parallel")

    start_time = time.time()
    try:
        batch_results = vectorized_processor(inputs={"text": batch_texts})
        vmap_time = time.time() - start_time
        print(f"Processing time: {vmap_time:.4f}s for {len(batch_texts)} items")

        # Inspect the output structure
        print(f"Output keys: {list(batch_results.keys())}")

        # Show processed results from each item
        for key, value in batch_results.items():
            if isinstance(value, list) and value:
                print(f"First '{key}' result: {value[0]}")
                print(f"Number of '{key}' results: {len(value)}")
    except Exception as e:
        print(f"Error in vmap processing: {e}")

    # 3. With pmap (parallel processing)
    print("\n--- Parallel Mapping (pmap) Example ---")
    print("pmap distributes processing across multiple workers.")
    print("Each worker processes one complete input independently.")

    # For pmap, we need a wrapper that can handle single texts properly
    class TextProcessorWrapper:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, *, inputs):
            """Process a batch of inputs serially, but with a parallel-compatible interface.

            When used with pmap, this wrapper ensures each worker gets a properly
            formatted single input.
            """
            # Check if we received the entire batch or a single item
            if "batch_index" in inputs and "text" in inputs:
                # We're in pmap context, each worker will get a single text at specific index
                batch_index = inputs["batch_index"]
                texts = inputs["text"]

                # Handle case where batch_index is a list (pmap dispatch)
                if isinstance(batch_index, list):
                    if batch_index:  # Non-empty list
                        batch_index = batch_index[0]  # Take first index
                    else:
                        return {"processed": "Empty batch index list"}

                # Now we have a scalar index
                if isinstance(batch_index, int) and 0 <= batch_index < len(texts):
                    # Process the text at our assigned index
                    return self.processor(inputs={"text": texts[batch_index]})
                else:
                    return {"processed": f"Error: Invalid batch index {batch_index}"}
            else:
                # Direct processing (not in pmap context)
                return self.processor(inputs=inputs)

    # Create wrapper and distribute processing
    wrapper = TextProcessorWrapper(processor)
    parallelized_wrapper = pmap(wrapper, num_workers=3)

    # Prepare parallel inputs with batch indices
    texts = ["Text for worker 1", "Text for worker 2", "Text for worker 3"]

    # Create inputs for each worker
    parallel_inputs = {
        "text": texts,
        "batch_index": list(range(len(texts))),  # Each worker gets their index
    }
    print(f"Input: {len(texts)} texts to be processed by different workers")

    start_time = time.time()
    try:
        parallel_results = parallelized_wrapper(inputs=parallel_inputs)
        pmap_time = time.time() - start_time
        print(f"Processing time: {pmap_time:.4f}s with {min(3, len(texts))} workers")

        # Inspect the output structure
        print(f"Output keys: {list(parallel_results.keys())}")

        # Show results from each worker
        for key, value in parallel_results.items():
            if isinstance(value, list) and value:
                print(f"First '{key}' result: {value[0]}")
                print(f"Number of '{key}' results: {len(value)}")
    except Exception as e:
        print(f"Error in pmap processing: {e}")


def compare_transform_combinations():
    """Compare different combinations of transforms for performance."""
    print("\n=== TRANSFORM COMBINATION COMPARISON ===")

    # Create base operators
    base_processor = TextProcessor(process_type="analyze", max_length=100)

    # Create a wrapper for pmap operations
    class TextProcessorWrapper:
        def __init__(self, processor):
            self.processor = processor

        def __call__(self, *, inputs):
            """Process a batch of inputs with a parallel-compatible interface."""
            try:
                # In pmap, inputs might contain both batch_index and text
                if "batch_index" in inputs and "text" in inputs:
                    batch_index = inputs["batch_index"]
                    texts = inputs["text"]

                    # Handle both scalar and list batch_index (pmap may distribute differently)
                    if isinstance(batch_index, list):
                        if batch_index:  # Non-empty list
                            batch_index = batch_index[0]  # Take first index
                        else:
                            return {"processed": "Empty batch index list"}

                    # Now we have a scalar index
                    if isinstance(batch_index, int) and 0 <= batch_index < len(texts):
                        selected_text = texts[batch_index]
                        # Handle item selection for strings or lists
                        if isinstance(selected_text, str):
                            return self.processor(inputs={"text": selected_text})
                        else:
                            # We might have gotten a nested structure
                            return {
                                "processed": f"Complex selected text: {type(selected_text)}"
                            }
                    else:
                        return {"processed": f"Invalid index {batch_index}"}

                # Direct text handling
                elif "text" in inputs:
                    if isinstance(inputs["text"], str):
                        # Process a single string directly
                        return self.processor(inputs=inputs)
                    elif isinstance(inputs["text"], list) and len(inputs["text"]) > 0:
                        # Process the first item from a list
                        return self.processor(inputs={"text": inputs["text"][0]})
                    else:
                        return {"processed": "Invalid text format"}
                else:
                    # No recognizable input format
                    return {"processed": "Missing text input"}

            except Exception as e:
                # Provide detailed error info
                return {"processed": f"Error: {str(e)}", "error": str(e)}

    # Create transformed variants
    vectorized = vmap(base_processor)
    wrapper = TextProcessorWrapper(base_processor)
    parallelized = pmap(wrapper, num_workers=4)
    # For combined transforms, we need to ensure interface compatibility
    vectorized_wrapper = TextProcessorWrapper(vectorized)
    vectorized_then_parallelized = pmap(vectorized_wrapper, num_workers=4)

    # Generate test dataset (100 items)
    texts = [
        f"Sample text number {i} for transformation comparison." for i in range(100)
    ]

    # Run with different configurations and measure performance
    results = []

    def time_execution(transform_name, operation_fn, data):
        """Time the execution of an operation and return results."""
        print(f"\nTesting: {transform_name}")

        try:
            start_time = time.time()
            result = operation_fn(inputs=data)
            elapsed = time.time() - start_time

            # Find the actual count of processed items by looking for list results
            count = 0
            for key, value in result.items():
                if isinstance(value, list):
                    count = max(count, len(value))

            # Default to 1 if no lists found
            if count == 0:
                count = 1

            throughput = count / max(elapsed, 0.0001)  # Avoid division by zero
            print(
                f"  Time: {elapsed:.4f}s for {count} items ({throughput:.2f} items/sec)"
            )
            return (transform_name, elapsed, count)
        except Exception as e:
            print(f"  Error: {e}")
            return (transform_name, float("inf"), 0)

    # 1. No transforms, sequential processing (properly matched interface)
    class SequentialProcessor:
        """Sequential processor that matches transform interface."""

        def __init__(self, processor):
            self.processor = processor

        def __call__(self, *, inputs):
            """Process inputs sequentially, matching transform interface.

            Args:
                inputs: Dict with text items, or list of texts

            Returns:
                Dict with processed results combined
            """
            if not isinstance(inputs, dict):
                # Handle direct list inputs
                texts = inputs if isinstance(inputs, list) else [inputs]
                inputs = {"text": texts}

            texts = inputs.get("text", [])
            if not isinstance(texts, list):
                texts = [texts]

            # Process each text individually
            results = {"processed": []}
            for text in texts:
                try:
                    result = self.processor(inputs={"text": text})
                    results["processed"].append(result.get("processed", ""))
                except Exception as e:
                    print(f"  Error processing item: {e}")
                    results["processed"].append(f"Error: {str(e)}")

            # Add other metadata
            results["process_type"] = self.processor.process_type
            results["original_length"] = [len(t) for t in texts]

            return results

    # Create sequential processor matching transform interface
    sequential_processor = SequentialProcessor(base_processor)

    # Benchmark sequential processing
    results.append(
        time_execution(
            "Sequential (no transforms)", sequential_processor, {"text": texts}
        )
    )

    # 2. vmap only (batch process all texts at once)
    print("\nExplaining vmap: Batch processing all inputs simultaneously")
    results.append(time_execution("vmap only", vectorized, {"text": texts}))

    # 3. pmap only (distribute texts across workers)
    print("\nExplaining pmap: Distributing inputs across parallel workers")
    # For pmap, provide indices for the workers
    pmap_inputs = {"text": texts, "batch_index": list(range(len(texts)))}
    results.append(time_execution("pmap only", parallelized, pmap_inputs))

    # 4. vmap + pmap (first batch, then distribute batches)
    print(
        "\nExplaining pmap(vmap()): Batching inputs, then distributing batches across workers"
    )
    # For combined transforms, we ensure interface compatibility
    # The wrapper expects batch_index
    pmap_vmap_inputs = {
        "text": texts,
        "batch_index": [0],  # We're sending the whole batch to one worker
    }
    results.append(
        time_execution("pmap(vmap())", vectorized_then_parallelized, pmap_vmap_inputs)
    )

    # 5. With execution options (controlling parallelism)
    print("\nDemonstrating execution_options: Controlling execution parameters")
    with execution_options(max_workers=2):
        results.append(
            time_execution(
                "pmap(vmap()) with max_workers=2",
                vectorized_then_parallelized,
                pmap_vmap_inputs,
            )
        )

    # Filter out failed executions
    valid_results = [
        (name, time_taken, count)
        for name, time_taken, count in results
        if time_taken != float("inf")
    ]

    if valid_results:
        # Find fastest execution
        fastest_result = min(valid_results, key=lambda x: x[1])
        fastest_time = fastest_result[1]

        # Display summary
        print("\nPerformance comparison (from slowest to fastest):")
        sorted_results = sorted(valid_results, key=lambda x: x[1], reverse=True)

        for name, time_taken, count in sorted_results:
            if time_taken > 0 and fastest_time > 0:
                speedup = time_taken / fastest_time
                print(
                    f"  {name}: {time_taken:.4f}s for {count} items ({speedup:.2f}x vs fastest)"
                )
            else:
                print(f"  {name}: {time_taken:.4f}s for {count} items (N/A)")
    else:
        print("\nNo valid performance results to compare.")


def demonstrate_structural_jit_with_vmap():
    """Demonstrate combining structural_jit with vmap."""
    print("\n=== STRUCTURAL JIT + VMAP EXAMPLE ===")

    # Create the optimized pipeline
    pipeline = TextAnalysisPipeline(process_type="analyze", feature_type="advanced")

    # Apply vmap to handle batches
    vectorized_pipeline = vmap(pipeline)

    # Generate a batch of inputs
    batch_texts = [
        "This is the first sample text for the JIT + vmap example.",
        "Here is a second sample with completely different content.",
        "A third sample text to ensure we have a good batch size.",
    ]

    # First run: Will include JIT compilation overhead
    print("First run (includes JIT compilation)...")
    start_time = time.time()
    results = vectorized_pipeline(inputs={"text": batch_texts})
    first_run_time = time.time() - start_time
    print(f"First run: {first_run_time:.4f}s for {len(batch_texts)} items")

    # Second run: Should be faster due to cached compilation
    print("\nSecond run (uses cached compilation)...")
    start_time = time.time()
    results = vectorized_pipeline(inputs={"text": batch_texts})
    second_run_time = time.time() - start_time
    print(f"Second run: {second_run_time:.4f}s for {len(batch_texts)} items")

    # Show improvement
    if first_run_time > 0:
        improvement = (first_run_time - second_run_time) / first_run_time * 100
        print(f"Improvement: {improvement:.1f}% faster with cached compilation")

    # Show feature results
    print("\nExtracted features for first item:")
    if results.get("features") and len(results["features"]) > 0:
        for feature_name, feature_value in results["features"][0][:3]:
            print(f"  {feature_name}: {feature_value:.4f}")

        if len(results["features"][0]) > 3:
            print(f"  ... and {len(results['features'][0]) - 3} more features")


def demonstrate_execution_options():
    """Demonstrate controlling transform behavior with execution options."""
    print("\n=== EXECUTION OPTIONS EXAMPLE ===")

    # Create a pipeline with both vmap and pmap
    base_pipeline = TextAnalysisPipeline(
        process_type="summarize",  # More expensive operation
        feature_type="comprehensive",
    )

    # First vectorize, then parallelize
    # This distributes batches across workers
    vectorized = vmap(base_pipeline)
    distributed = pmap(vectorized, num_workers=4)

    # Generate larger dataset (50 items)
    texts = [
        f"Sample document {i} with content for execution options testing."
        for i in range(50)
    ]
    data = {"text": texts}

    # Different execution options configurations
    configurations = [
        ("Default settings", {}),
        ("Sequential execution", {"use_parallel": False}),
        ("Limited workers", {"max_workers": 2}),
        ("With caching", {"enable_caching": True}),
        ("Sequential + caching", {"use_parallel": False, "enable_caching": True}),
        (
            "Full parallel + caching",
            {"use_parallel": True, "max_workers": 8, "enable_caching": True},
        ),
    ]

    # Test each configuration
    results = []

    for name, options in configurations:
        print(f"\nTesting: {name}")

        # Apply the execution options
        with execution_options(**options):
            start_time = time.time()
            result = distributed(inputs=data)
            elapsed = time.time() - start_time

            count = len(result.get("text_length", []))
            print(
                f"  Processed {count} items in {elapsed:.4f}s ({count/elapsed:.2f} items/sec)"
            )

            results.append((name, elapsed, count))

    # Show comparative summary
    print("\nExecution options performance comparison:")
    sorted_results = sorted(results, key=lambda x: x[1])

    for name, time_taken, count in sorted_results:
        speedup = results[0][1] / time_taken if time_taken > 0 else float("inf")
        print(f"  {name}: {time_taken:.4f}s ({speedup:.2f}x vs default)")


def demonstrate_chunked_processing():
    """Demonstrate memory-efficient chunked processing for large datasets."""
    print("\n=== CHUNKED PROCESSING EXAMPLE ===")

    # Create a simpler minimal processor specifically for this example
    # This avoids the complexity of the JIT integration
    class SimpleTextProcessor:
        """Simple text processor for chunked processing demo."""

        def __init__(self):
            """Initialize the processor."""
            pass

        def __call__(self, *, inputs):
            """Process a text input to demonstrate chunking."""
            text = inputs.get("text", "")
            if isinstance(text, str):
                # Process one text item
                return {"processed": f"Processed: {text[:20]}...", "length": len(text)}
            else:
                # Return empty for non-string inputs
                return {"processed": "Invalid input", "length": 0}

    # Create the processor and vectorized version
    processor = SimpleTextProcessor()

    # Create a simple vectorized version
    def batch_process(*, inputs):
        """Simple batch processor that mimics vmap behavior."""
        texts = inputs.get("text", [])
        if not isinstance(texts, list):
            texts = [texts]

        results = []
        for text in texts:
            results.append(processor(inputs={"text": text}))

        # Combine results by key (similar to vmap)
        combined = {}
        for key in ["processed", "length"]:
            combined[key] = [r.get(key, "") for r in results]

        return combined

    # Generate "large" dataset that we'll process in chunks
    large_dataset = [
        f"Document {i} with content for chunked processing example." for i in range(200)
    ]

    print(
        "This example demonstrates processing a large dataset in chunks to manage memory."
    )
    print(
        "For large language model processing, this pattern helps avoid GPU out-of-memory errors."
    )

    def process_in_chunks(texts, chunk_size=20):
        """Process a dataset in chunks to control memory usage.

        This demonstrates how to efficiently process large datasets by breaking
        them into manageable chunks, avoiding memory issues.

        Args:
            texts: List of texts to process
            chunk_size: Number of items to process in each chunk

        Returns:
            Combined results from all chunks
        """
        # Initialize results container
        all_results = {"processed": [], "length": []}

        # Get total size
        total_items = len(texts)
        chunk_count = (total_items + chunk_size - 1) // chunk_size

        print(
            f"Processing {total_items} items in {chunk_count} chunks of size {chunk_size}..."
        )

        # Process in chunks
        total_processed = 0
        for i in range(0, total_items, chunk_size):
            # Create chunk
            end_idx = min(i + chunk_size, total_items)
            chunk = texts[i:end_idx]
            current_chunk_size = len(chunk)

            # Process chunk
            chunk_start = time.time()
            try:
                chunk_results = batch_process(inputs={"text": chunk})
                chunk_elapsed = time.time() - chunk_start

                # Calculate throughput (with safety for very fast execution)
                throughput = current_chunk_size / max(chunk_elapsed, 0.0001)

                print(
                    f"  Chunk {i//chunk_size + 1}/{chunk_count}: "
                    f"{current_chunk_size} items in {chunk_elapsed:.4f}s "
                    f"({throughput:.2f} items/sec)"
                )

                # Append results
                for key in all_results:
                    if key in chunk_results and isinstance(chunk_results[key], list):
                        all_results[key].extend(chunk_results[key])
                        total_processed += len(chunk_results[key])

            except Exception as e:
                print(f"  Error processing chunk {i//chunk_size + 1}: {e}")

        print(f"Successfully processed {total_processed} items in chunks")
        return all_results

    # Process with chunking and measure total time
    start_time = time.time()
    results = process_in_chunks(large_dataset, chunk_size=20)
    total_elapsed = max(time.time() - start_time, 0.0001)  # Avoid div by zero

    print(
        f"\nTotal processing time: {total_elapsed:.4f}s for {len(large_dataset)} items"
    )
    print(f"Average processing rate: {len(large_dataset)/total_elapsed:.2f} items/sec")
    print(f"Results count: {len(results['processed'])} items processed")

    # Show sample results
    if results["processed"]:
        print(f"\nSample result: {results['processed'][0]}")
        print(f"Total bytes processed: {sum(results['length'])}")
    else:
        print("\nNo processed results found")

    # Compare with non-chunked (if small enough to run all at once)
    if len(large_dataset) <= 50:  # Only do this comparison for smaller datasets
        print("\nComparing to non-chunked processing:")
        start_time = time.time()
        full_results = batch_process(
            inputs={"text": large_dataset[:10]}
        )  # Just use 10 items for the test
        full_elapsed = max(time.time() - start_time, 0.0001)  # Avoid div by zero
        print(
            f"Non-chunked processing: {full_elapsed:.4f}s ({10/full_elapsed:.2f} items/sec)"
        )


###############################################################################
# Main Demonstration
###############################################################################


def main():
    """Run the transforms integration demonstrations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Ember XCS Transforms Integration Example")
    print("=======================================")
    print("This example demonstrates how to combine vmap, pmap, and structural_jit")
    print("transforms with execution options for optimized processing pipelines.")

    # Run the demonstrations
    demonstrate_basic_transforms()
    compare_transform_combinations()
    demonstrate_structural_jit_with_vmap()
    demonstrate_execution_options()
    demonstrate_chunked_processing()

    print("\n=== KEY TAKEAWAYS ===")
    print("1. vmap is best for batch processing of homogeneous data")
    print("2. pmap works well for distributing work across CPU cores")
    print("3. structural_jit optimizes complex operator composition")
    print("4. pmap(vmap()) is typically more efficient than vmap(pmap())")
    print("5. execution_options provide fine-grained control over execution behavior")
    print("6. Chunked processing helps manage memory for large datasets")
    print("7. JIT compilation has upfront cost but amortizes over repeated calls")

    print("\nFor more information, see:")
    print("- docs/xcs/TRANSFORMS.md")
    print("- docs/xcs/EXECUTION_OPTIONS.md")
    print("- docs/xcs/JIT_OVERVIEW.md")


if __name__ == "__main__":
    main()
