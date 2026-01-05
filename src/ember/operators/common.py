"""Common operators built on the simplified base class.

This module provides commonly used operators that demonstrate the power of
the simplified design. These operators showcase composition, ensemble patterns,
and integration with language models.
"""

import importlib
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Sequence

from ember.api.models import ModelBinding, Response, models
from ember.operators.base import Operator


class ModelCall(Operator):
    """EXPERIMENTAL: Convenience operator that calls a language model.

    Helpful for operator chaining.
    Note: This is a convenience wrapper, not a performance feature.
    JIT/vmap do not provide speedups for network-bound LLM calls; for
    production-critical paths prefer calling `models()` directly.

    This operator wraps a model binding and provides a consistent interface
    for calling language models. It returns the complete response object,
    preserving metadata like token counts, costs, and model information.

    Attributes:
        model: The bound model instance to call.

    Examples:
        Basic model calling:

        >>> # Create operator with default model
        >>> model_op = ModelCall()
        >>> response = model_op("What is the capital of France?")
        >>> print(response.text)  # "Paris is the capital of France."
        >>> print(response.usage["total_tokens"])  # 25

        >>> # Use specific model
        >>> claude_op = ModelCall("claude-3-sonnet")
        >>> response = claude_op("Explain quantum computing")
        >>> print(f"Cost: ${response.usage['cost']:.4f}")

        >>> # Chain with text extraction
        >>> from ember.api.decorators import op
        >>> @op
        >>> def extract_text(response):
        ...     return response.text
        >>>
        >>> pipeline = Chain([
        ...     ModelCall("gpt-4"),
        ...     extract_text
        ... ])
        >>> result = pipeline("Summarize this article")
    """

    model: ModelBinding

    def __init__(self, model_name: str = "gpt-4o", **kwargs: Any):
        """Initialize model call operator.

        Args:
            model_name: Name of the model to use (e.g., "gpt-4", "claude-3").
            **kwargs: Additional arguments passed to model initialization.
        """
        self.model = models.instance(model_name, **kwargs)

    def forward(self, input: Any) -> Response:
        """Call the model with the input and return full response.

        Args:
            input: The input to send to the model.

        Returns:
            Complete response object with text, metadata, usage, and costs.
        """
        return self.model.response(str(input))


class Ensemble(Operator):
    """Ensemble operator that combines multiple operators.

    This operator runs multiple sub-operators and aggregates their results.
    Execution is sequential in the current implementation. For network-bound
    workloads where parallelism is desired, a separate concurrent variant may
    be introduced to avoid coupling concerns here.

    Attributes:
        operators: List of operators to run in ensemble.
        aggregator: Optional function to combine results. If None, returns list.

    Examples:
        Simple ensemble returning all results:

        >>> ensemble = Ensemble([
        ...     Classifier("gpt-4"),
        ...     Classifier("claude-3"),
        ...     Classifier("gemini")
        ... ])
        >>> results = ensemble("Is this text positive?")
        >>> # results = ["positive", "positive", "neutral"]

        Ensemble with custom aggregation:

        >>> def majority_vote(results: List[str]) -> str:
        ...     from collections import Counter
        ...     return Counter(results).most_common(1)[0][0]
        >>>
        >>> ensemble = Ensemble(
        ...     operators=[Op1(), Op2(), Op3()],
        ...     aggregator=majority_vote
        ... )
        >>> result = ensemble(input_data)  # Returns single majority result
    """

    operators: Sequence[Operator]
    aggregator: Optional[Callable[[List[Any]], Any]]

    def __init__(
        self,
        operators: Sequence[Operator],
        aggregator: Optional[Callable[[List[Any]], Any]] = None,
    ):
        """Initialize ensemble with operators and optional aggregator.

        Args:
            operators: List of operators to run in ensemble.
            aggregator: Optional function to aggregate results. Should accept
                a list of results and return aggregated result.
        """
        self.operators = operators
        self.aggregator = aggregator

    def forward(self, input: Any) -> Any:
        """Run all operators and aggregate results.

        Args:
            input: Input to pass to all operators.

        Returns:
            List of results if no aggregator, otherwise aggregated result.
        """
        results = [op(input) for op in self.operators]

        if self.aggregator:
            return self.aggregator(results)
        return results


class Chain(Operator):
    """Chain operator that runs operators sequentially.

    This operator passes the output of each operator as input to the next,
    creating a pipeline of transformations. Useful for multi-step processing.

    Examples:
        >>> chain = Chain([
        ...     Preprocessor(),      # Clean and normalize text
        ...     Classifier(),        # Classify cleaned text
        ...     Postprocessor()      # Format classification result
        ... ])
        >>> result = chain(raw_input)
    """

    operators: Sequence[Operator]

    def __init__(self, operators: Sequence[Operator]):
        """Initialize chain with list of operators.

        Args:
            operators: List of operators to run in sequence.
        """
        self.operators = operators

    def forward(self, input: Any) -> Any:
        """Pass input through all operators sequentially.

        Args:
            input: Initial input to the chain.

        Returns:
            Output from the final operator.
        """
        result = input
        for op in self.operators:
            result = op(result)
        return result


class Router(Operator):
    """Router that conditionally directs inputs to different operators.

    This operator uses a routing function to decide which operator should
    handle each input. Useful for specialization and conditional logic.

    Examples:
        Text type router:

        >>> def route_by_length(text: str) -> str:
        ...     return "long" if len(text) > 100 else "short"
        >>>
        >>> router = Router(
        ...     routes={
        ...         "short": ShortTextProcessor(),
        ...         "long": LongTextProcessor()
        ...     },
        ...     router_fn=route_by_length
        ... )

        Domain-specific router:

        >>> def route_by_domain(question: str) -> str:
        ...     if "math" in question.lower():
        ...         return "math"
        ...     elif "code" in question.lower():
        ...         return "code"
        ...     else:
        ...         return "general"
        >>>
        >>> router = Router(
        ...     routes={
        ...         "math": MathExpert(),
        ...         "code": CodeExpert(),
        ...         "general": GeneralAssistant()
        ...     },
        ...     router_fn=route_by_domain
        ... )
    """

    routes: Dict[str, Operator]
    router_fn: Callable[[Any], str]
    default_route: Optional[str]

    def __init__(
        self,
        routes: Dict[str, Operator],
        router_fn: Callable[[Any], str],
        default_route: Optional[str] = None,
    ):
        """Initialize router with routes and routing function.

        Args:
            routes: Dictionary mapping route names to operators.
            router_fn: Function that takes input and returns route name.
            default_route: Optional default route if router_fn returns unknown.
        """
        self.routes = routes
        self.router_fn = router_fn
        self.default_route = default_route

    def forward(self, input: Any) -> Any:
        """Route input to appropriate operator.

        Args:
            input: Input to route.

        Returns:
            Result from the selected operator.

        Raises:
            KeyError: If route not found and no default route.
        """
        route = self.router_fn(input)

        if route not in self.routes:
            if self.default_route and self.default_route in self.routes:
                route = self.default_route
            else:
                raise KeyError(f"No operator for route '{route}'")

        return self.routes[route](input)


def _import_jax() -> tuple[Any, Any]:
    """Import JAX modules. Raises RuntimeError if JAX not installed."""
    try:
        jax_module = importlib.import_module("jax")
        jnp_module = importlib.import_module("jax.numpy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("LearnableRouter requires JAX to be installed.") from exc
    return jax_module, jnp_module


class LearnableRouter(Operator):
    """Router with learnable routing logic using JAX.

    Routes inputs based on embeddings and learnable weights. The routing
    decision is differentiable, enabling end-to-end optimization.

    With embedding_fn: computes embeddings from input automatically.
    Without embedding_fn: expects input with 'data' and 'embedding' attributes.
    """

    routes: Dict[str, Operator]
    route_names: List[str]
    embedding_fn: Optional[Callable[[Any], Any]]
    routing_weights: Any  # jax.Array - typed as Any for optional JAX
    temperature: Any  # jax.Array

    def __init__(
        self,
        routes: Dict[str, Operator],
        embed_dim: int,
        key: Any,  # jax.random.PRNGKey
        embedding_fn: Optional[Callable[[Any], Any]] = None,
        temperature: float = 1.0,
    ):
        """Initialize learnable router.

        Args:
            routes: Route names mapped to operators.
            embed_dim: Dimension of input embeddings.
            key: JAX random key for weight initialization.
            embedding_fn: Function to compute embeddings, or None to use input.embedding.
            temperature: Softmax temperature (lower = more decisive).

        Raises:
            ValueError: If temperature <= 0.
        """
        jax, jnp = _import_jax()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.routes = routes
        self.route_names = list(routes.keys())
        self.embedding_fn = embedding_fn
        self.routing_weights = jax.random.normal(key, (embed_dim, len(routes)))
        self.temperature = jnp.array(float(temperature))

    def compute_route_probabilities(self, embedding: Any) -> Any:
        """Compute routing probabilities from embedding (differentiable)."""
        jax, _ = _import_jax()
        logits = embedding @ self.routing_weights
        return jax.nn.softmax(logits / self.temperature)

    def forward(self, input: Any) -> Any:
        """Route input based on learned weights."""
        jax, jnp = _import_jax()

        if self.embedding_fn is not None:
            data = input
            embedding = self.embedding_fn(input)
        elif hasattr(input, "data") and hasattr(input, "embedding"):
            data = input.data
            embedding = input.embedding
        else:
            raise ValueError(
                "When embedding_fn is None, input must have 'data' and 'embedding' attributes."
            )

        probs = self.compute_route_probabilities(embedding)
        route_idx = int(jax.device_get(jnp.argmax(probs)))
        return self.routes[self.route_names[route_idx]](data)


class Retry(Operator):
    """Operator that retries on failure with exponential backoff and jitter.

    Examples:
        >>> reliable_op = Retry(UnreliableAPIOperator(), max_attempts=3)
        >>> careful_op = Retry(
        ...     RateLimitedOperator(),
        ...     should_retry=lambda e, n: isinstance(e, RateLimitError),
        ...     base_delay=0.5,
        ... )
    """

    operator: Operator
    max_attempts: int
    base_delay: float
    max_delay: float
    should_retry: Callable[[Exception, int], bool]

    def __init__(
        self,
        operator: Operator,
        max_attempts: int = 3,
        should_retry: Optional[Callable[[Exception, int], bool]] = None,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.operator = operator
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.should_retry = should_retry or (lambda e, n: n < max_attempts)

    def _compute_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter: base * 2^(attempt-1) + random jitter."""
        import random

        delay: float = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        jitter: float = random.uniform(0, delay * 0.5)
        return delay + jitter

    def forward(self, input: Any) -> Any:
        import time

        last_error: Exception

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.operator(input)
            except Exception as e:
                last_error = e
                if not self.should_retry(e, attempt):
                    raise
                if attempt < self.max_attempts:
                    time.sleep(self._compute_delay(attempt))

        raise last_error


class Cache(Operator):
    """LRU cache wrapper for expensive operators. Not JAX-compatible (mutable state).

    Examples:
        >>> cached = Cache(ExpensiveClassifier(), max_size=1000)
        >>> result1 = cached("input")  # computes
        >>> result2 = cached("input")  # cache hit
    """

    operator: Operator
    max_size: int
    key_fn: Callable[[Any], str]
    _cache: OrderedDict[str, Any]

    def __init__(
        self,
        operator: Operator,
        max_size: int = 100,
        key_fn: Optional[Callable[[Any], str]] = None,
    ):
        self.operator = operator
        self.max_size = max_size
        self.key_fn = key_fn or str
        self._cache = OrderedDict()

    def forward(self, input: Any) -> Any:
        key = self.key_fn(input)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        result = self.operator(input)
        self._cache[key] = result

        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

        return result


# Convenience operators for common model calling tasks


class ExtractText(Operator):
    """EXPERIMENTAL: Operator that extracts text from a model response.

    WARNING: This operator is somewhat redundant, but helpful for chaining.

    This operator takes a Response object and returns just the text content;
    useful for chaining after model calls when you only need the text output.
    """

    def forward(self, response: Response) -> str:
        """Return the text from a ``Response``.

        Args:
            response: The model ``Response`` object to extract text from.

        Returns:
            The ``text`` field contained in the response.
        """
        return response.text


class ModelText(Operator):
    """EXPERIMENTAL: Convenience operator that returns only the text.

    Combines ModelCall and ExtractText for cleaner chains. This is a
    convenience wrapper; use `models()` directly for production.
    """

    model_text: Operator

    def __init__(self, model_name: str, **kwargs: Any):
        """Initialize the model-text operator.

        Args:
            model_name: Name of the model to use (for example, ``"gpt-4o"``).
            **kwargs: Additional keyword arguments forwarded to
                ``ModelCall``/``models.instance``.
        """
        self.model_text = Chain([ModelCall(model_name, **kwargs), ExtractText()])

    def forward(self, input: Any) -> str:
        """Call the model and return only the text content.

        Args:
            input: The input to send to the underlying model.

        Returns:
            The text content from the model's response.
        """
        # Cast to str since we know ExtractText returns str
        return str(self.model_text(input))


# Convenience functions for creating common patterns


def ensemble(*operators: Operator | Sequence[Operator], **kwargs: Any) -> Ensemble:
    """Create an ensemble from operators.

    Args:
        *operators: Operators to include in ensemble. Can be individual operators
                   or a single list/tuple of operators.
        **kwargs: Additional arguments for Ensemble constructor.

    Returns:
        Ensemble operator.

    Examples:
        >>> # Individual operators
        >>> ensemble_op = ensemble(op1, op2, op3)
        >>>
        >>> # List of operators
        >>> ops = [op1, op2, op3]
        >>> ensemble_op = ensemble(ops)
    """
    # Handle both individual operators and a list
    if len(operators) == 1 and isinstance(operators[0], (list, tuple)):
        return Ensemble(list(operators[0]), **kwargs)

    # Cast tuple[Operator | Sequence[Operator], ...] to list[Operator]
    # We filtered out the sequence case above, so rest are operators
    return Ensemble(list(operators), **kwargs)  # type: ignore[arg-type]


def chain(*operators: Operator | Sequence[Operator]) -> Chain:
    """Create a chain from operators.

    Args:
        *operators: Operators to chain in sequence.

    Returns:
        Chain operator.
    """
    if len(operators) == 1 and isinstance(operators[0], (list, tuple)):
        return Chain(list(operators[0]))
    return Chain(list(operators))  # type: ignore[arg-type]


def router(routes: Dict[str, Operator], **kwargs: Any) -> Router:
    """Create a router from route dictionary.

    Args:
        routes: Dictionary mapping route names to operators.
        **kwargs: Additional arguments for Router constructor.

    Returns:
        Router operator.
    """
    return Router(routes, **kwargs)
