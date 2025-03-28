"""Evaluation API for Ember.

This module provides tools for evaluating model outputs against reference answers
and computing performance metrics. It supports both standard evaluators and custom
evaluation functions.

Standard Evaluators:
    - "exact_match": Checks for exact match between prediction and reference 
      (case-insensitive)
    - "accuracy": Alias for "exact_match"
    - "numeric": Compares numeric values with customizable tolerance
    - "regex": Extracts content using regex patterns and evaluates the extracted text

Examples:
    # List available evaluators in the registry
    from ember.api import eval
    available = eval.list_available_evaluators()
    print(f"Available evaluators: {available}")
    
    # Using standard evaluators from the registry
    accuracy = eval.Evaluator.from_registry("exact_match")
    
    # Evaluators with parameters
    numeric = eval.Evaluator.from_registry("numeric", tolerance=0.01)
    regex = eval.Evaluator.from_registry("regex", pattern=r"answer: (.*)")
    
    # Creating a custom evaluator function
    def custom_metric(prediction, reference):
        return {
            "is_correct": prediction.lower() == reference.lower(),
            "char_count": len(prediction)
        }
    
    custom_eval = eval.Evaluator.from_function(custom_metric)
    
    # Evaluating a model on a dataset
    from ember.api import datasets, models
    
    test_data = datasets("mmlu").subset("physics").split("test")
    model = models.openai.gpt4o
    
    pipeline = eval.EvaluationPipeline([accuracy, custom_eval])
    results = pipeline.evaluate(model, test_data)
    
    print(f"Accuracy: {results['is_correct']*100:.1f}%")
    print(f"Average character count: {results['char_count']:.1f}")
    
Custom Evaluator Implementation:
    # Implementing a custom evaluator class directly
    from ember.api.eval import IEvaluator, EvaluationResult
    
    class WordCountEvaluator(IEvaluator):
        def evaluate(self, system_output, correct_answer, **kwargs):
            word_count = len(system_output.split())
            is_correct = word_count >= kwargs.get("min_words", 5)
            return EvaluationResult(
                is_correct=is_correct,
                score=min(1.0, word_count / 20),  # Normalize to 0-1
                metadata={"word_count": word_count}
            )
    
    # Register custom evaluator in registry
    eval.register_evaluator("word_count", WordCountEvaluator)
    
    # Use registered custom evaluator
    word_counter = eval.Evaluator.from_registry("word_count", min_words=10)
"""

from typing import Any, Callable, Dict

from ember.core.utils.eval.base_evaluator import EvaluationResult, IEvaluator
from ember.core.utils.eval.registry import EvaluatorRegistry

# Initialize registry with standard evaluators
_REGISTRY = EvaluatorRegistry()


# Register standard evaluators
def _init_registry():
    """Initialize the evaluator registry with standard evaluators."""
    from ember.core.utils.eval.evaluators import (
        ExactMatchEvaluator,
        NumericToleranceEvaluator,
        PartialRegexEvaluator,
    )

    # Skip if already initialized
    if hasattr(_REGISTRY, "_initialized"):
        return

    # Register basic evaluators with standard names
    _REGISTRY.register("exact_match", ExactMatchEvaluator)
    _REGISTRY.register("accuracy", ExactMatchEvaluator)  # Alias for exact_match
    _REGISTRY.register("numeric", lambda **kwargs: NumericToleranceEvaluator(**kwargs))
    _REGISTRY.register(
        "regex", lambda pattern, **kwargs: PartialRegexEvaluator(pattern=pattern)
    )

    # Mark as initialized
    _REGISTRY._initialized = True


# Initialize the registry
_init_registry()


def list_available_evaluators() -> list[str]:
    """Returns a list of available evaluator names in the registry.

    Returns:
        List of evaluator names that can be used with Evaluator.from_registry()

    Example:
        ```python
        available = eval.list_available_evaluators()
        print(f"Available evaluators: {available}")
        ```
    """
    # Ensure registry is initialized
    _init_registry()

    # Get all keys from the registry's internal dictionary
    return sorted(list(_REGISTRY._registry.keys()))


def register_evaluator(name: str, evaluator_factory: Callable[..., IEvaluator]) -> None:
    """Registers a new evaluator in the global registry.

    Args:
        name: Unique identifier for the evaluator
        evaluator_factory: Factory function or class that returns an IEvaluator instance

    Example:
        ```python
        class MyEvaluator(IEvaluator):
            def evaluate(self, system_output, correct_answer, **kwargs):
                return EvaluationResult(is_correct=True, score=1.0)

        register_evaluator("my_evaluator", MyEvaluator)
        ```
    """
    # Ensure registry is initialized
    _init_registry()

    # Register the evaluator
    _REGISTRY.register(name, evaluator_factory)


class Evaluator:
    """Evaluator for model outputs against reference answers.

    Provides a unified interface for evaluating model predictions against
    reference answers,
    with support for registry-based and custom function-based evaluators.
    """

    @classmethod
    def from_registry(cls, name: str, **kwargs) -> "Evaluator":
        """Create an evaluator from the registry.

        Args:
            name: Name of registered evaluator
            **kwargs: Additional configuration for the evaluator

        Returns:
            An initialized Evaluator

        Raises:
            KeyError: If no evaluator with the given name exists in the registry
        """
        evaluator = _REGISTRY.create(name, **kwargs)
        return cls(evaluator=evaluator)

    @classmethod
    def from_function(cls, func: Callable[[Any, Any], Dict[str, Any]]) -> "Evaluator":
        """Create an evaluator from a custom function.

        Args:
            func: Function taking (prediction, reference) returning metric dict

        Returns:
            An initialized Evaluator
        """

        # Create a function adapter that conforms to IEvaluator interface
        class FunctionAdapter(IEvaluator):
            def __init__(self, func: Callable[[Any, Any], Dict[str, Any]]) -> None:
                self.func = func

            def evaluate(
                self, system_output: Any, correct_answer: Any, **kwargs: Any
            ) -> EvaluationResult:
                # Call the function to get metric dict
                metrics = self.func(system_output, correct_answer, **kwargs)

                # Extract standard metrics if present, or use defaults
                is_correct = metrics.get("is_correct", metrics.get("correct", False))
                score = metrics.get("score", float(is_correct))

                # Return proper evaluation result
                return EvaluationResult(
                    is_correct=is_correct, score=score, metadata=metrics
                )

        return cls(evaluator=FunctionAdapter(func))

    def __init__(self, evaluator: IEvaluator) -> None:
        """Initialize with an evaluator implementation.

        Args:
            evaluator: An object with an evaluate method
        """
        self.evaluator = evaluator

    def evaluate(
        self, prediction: Any, reference: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Evaluate a prediction against a reference.

        Args:
            prediction: The model's prediction or output
            reference: The expected answer or reference
            **kwargs: Additional parameters for evaluation

        Returns:
            A dictionary of evaluation metrics
        """
        result = self.evaluator.evaluate(prediction, reference, **kwargs)

        # Extract metrics from the result
        metrics: Dict[str, Any] = {}
        if hasattr(result, "is_correct"):
            metrics["is_correct"] = result.is_correct
        if hasattr(result, "score"):
            metrics["score"] = result.score
        if hasattr(result, "metadata") and result.metadata:
            metrics.update(result.metadata)

        return metrics


class EvaluationPipeline:
    """Pipeline for evaluating models on datasets.

    Combines multiple evaluators and applies them to model outputs on a dataset,
    aggregating the results into a comprehensive evaluation report.
    """

    def __init__(self, evaluators: list[Evaluator]) -> None:
        """Initialize with a list of evaluators.

        Args:
            evaluators: List of Evaluator instances to apply
        """
        self.evaluators = evaluators

    def evaluate(self, model: Any, dataset: Any) -> Dict[str, float]:
        """Evaluate a model on a dataset, returning aggregated metrics.

        Args:
            model: A callable model or operator that produces predictions
            dataset: A dataset containing input examples and references

        Returns:
            Dictionary of aggregated evaluation metrics
        """
        # Initialize metric collectors
        all_metrics: Dict[str, list] = {}
        processed_count = 0
        error_count = 0

        # Setup logging
        import logging

        logger = logging.getLogger("ember.eval")

        # Process each example in the dataset
        for item in dataset:
            # Extract input and reference
            input_text = None
            reference = None

            # Extract content from item, supporting different dataset formats
            # Handle item.content or direct item dictionary
            has_content = hasattr(item, "content")
            content = item.content if has_content else item

            # Try different commonly used field names for input
            for input_field in ["query", "prompt", "input", "question"]:
                if isinstance(content, dict) and input_field in content:
                    input_text = content[input_field]
                    break

            # Try different commonly used field names for reference
            for ref_field in ["answer", "reference", "output", "ground_truth"]:
                if isinstance(content, dict) and ref_field in content:
                    reference = content[ref_field]
                    break

            # Skip items without required fields
            if input_text is None or reference is None:
                if isinstance(content, dict):
                    keys = ", ".join(sorted(content.keys()))
                    logger.debug(
                        f"Skipping item, missing required fields. Keys: {keys}"
                    )
                else:
                    logger.debug("Skipping item, content is not a dictionary")
                continue

            # Get model prediction
            try:
                if callable(model):
                    # Function-like model
                    prediction = model(input_text)
                else:
                    # Operator-like model
                    prediction = model(inputs={"query": input_text})

                processed_count += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"Error getting prediction: {e}")
                continue

            # Apply all evaluators
            for evaluator in self.evaluators:
                try:
                    metrics = evaluator.evaluate(prediction, reference)

                    # Aggregate metrics
                    for key, value in metrics.items():
                        if key not in all_metrics:
                            all_metrics[key] = []
                        if isinstance(value, (int, float)):
                            all_metrics[key].append(value)
                except Exception as e:
                    logger.warning(f"Error in evaluator {evaluator}: {e}")

        # Calculate average metrics
        results: Dict[str, float] = {}
        for k, v in all_metrics.items():
            if v:  # Non-empty list
                results[k] = sum(v) / len(v)
            else:
                results[k] = 0.0

        # Add summary metrics
        if processed_count > 0:
            results["processed_count"] = processed_count
            # Calculate error rate safely
            results["error_rate"] = (
                error_count / processed_count if processed_count > 0 else 0.0
            )

        return results


__all__ = [
    # Core evaluation classes
    "Evaluator",
    "EvaluationPipeline",
    # Registry functions
    "list_available_evaluators",
    "register_evaluator",
    # Core implementation re-exports
    "EvaluationResult",
    "IEvaluator",
]
