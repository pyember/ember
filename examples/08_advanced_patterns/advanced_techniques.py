"""Advanced techniques for building sophisticated AI systems.

This example demonstrates:
- Dynamic operator composition
- Meta-programming patterns
- Plugin architectures
- Advanced configuration patterns

Run with:
    python examples/08_advanced_patterns/advanced_techniques.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Type

from ember.api import op


# =============================================================================
# Part 1: Dynamic Operator Composition
# =============================================================================

@dataclass
class OperatorSpec:
    """Specification for creating operators dynamically."""

    name: str
    transform: Callable[[Any], Any]
    input_validator: Optional[Callable[[Any], bool]] = None
    output_validator: Optional[Callable[[Any], bool]] = None


class OperatorFactory:
    """Factory for creating operators from specifications."""

    def __init__(self):
        self._registry: Dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec) -> None:
        """Register an operator specification."""
        self._registry[spec.name] = spec

    def create(self, name: str) -> Callable[[Any], Any]:
        """Create an operator from a registered spec."""
        spec = self._registry.get(name)
        if not spec:
            raise KeyError(f"Unknown operator: {name}")

        @op
        def dynamic_op(data: Any) -> Any:
            if spec.input_validator and not spec.input_validator(data):
                raise ValueError(f"Input validation failed for {name}")

            result = spec.transform(data)

            if spec.output_validator and not spec.output_validator(result):
                raise ValueError(f"Output validation failed for {name}")

            return result

        return dynamic_op

    def compose(self, *names: str) -> Callable[[Any], Any]:
        """Compose multiple operators into a pipeline."""
        operators = [self.create(name) for name in names]

        @op
        def composed(data: Any) -> Any:
            result = data
            for operator in operators:
                result = operator(result)
            return result

        return composed


def demonstrate_dynamic_composition() -> None:
    """Show dynamic operator composition."""
    print("Part 1: Dynamic Operator Composition")
    print("-" * 50)

    factory = OperatorFactory()

    # Register operators
    factory.register(OperatorSpec(
        name="uppercase",
        transform=lambda x: x.upper() if isinstance(x, str) else x,
    ))
    factory.register(OperatorSpec(
        name="strip",
        transform=lambda x: x.strip() if isinstance(x, str) else x,
    ))
    factory.register(OperatorSpec(
        name="prefix",
        transform=lambda x: f"[PROCESSED] {x}",
    ))

    # Create individual operators
    uppercase = factory.create("uppercase")
    print(f"uppercase('hello'): {uppercase('hello')}")

    # Compose pipeline
    pipeline = factory.compose("strip", "uppercase", "prefix")
    result = pipeline("  hello world  ")
    print(f"Pipeline result: {result}")
    print()


# =============================================================================
# Part 2: Plugin Architecture
# =============================================================================

class Plugin(Protocol):
    """Protocol for plugins."""

    name: str

    def initialize(self) -> None:
        """Initialize the plugin."""
        ...

    def process(self, data: Any) -> Any:
        """Process data through the plugin."""
        ...


@dataclass
class PluginManager:
    """Manages plugin registration and execution."""

    plugins: Dict[str, Plugin] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)

    def register(self, plugin: Plugin) -> None:
        """Register a plugin."""
        self.plugins[plugin.name] = plugin
        if plugin.name not in self.execution_order:
            self.execution_order.append(plugin.name)

    def initialize_all(self) -> None:
        """Initialize all registered plugins."""
        for name in self.execution_order:
            self.plugins[name].initialize()

    def execute_pipeline(self, data: Any) -> Any:
        """Execute all plugins in order."""
        result = data
        for name in self.execution_order:
            result = self.plugins[name].process(result)
        return result


class LoggingPlugin:
    """Plugin that logs data flow."""

    name = "logging"

    def initialize(self) -> None:
        print("  LoggingPlugin initialized")

    def process(self, data: Any) -> Any:
        print(f"  [LOG] Processing: {str(data)[:50]}")
        return data


class TransformPlugin:
    """Plugin that transforms data."""

    name = "transform"

    def initialize(self) -> None:
        print("  TransformPlugin initialized")

    def process(self, data: Any) -> Any:
        if isinstance(data, str):
            return {"text": data, "length": len(data)}
        return data


def demonstrate_plugin_architecture() -> None:
    """Show plugin architecture pattern."""
    print("Part 2: Plugin Architecture")
    print("-" * 50)

    manager = PluginManager()

    # Register plugins
    manager.register(LoggingPlugin())
    manager.register(TransformPlugin())

    # Initialize
    print("Initializing plugins:")
    manager.initialize_all()
    print()

    # Execute
    print("Executing pipeline:")
    result = manager.execute_pipeline("Hello, World!")
    print(f"Final result: {result}")
    print()


# =============================================================================
# Part 3: Strategy Pattern for Algorithms
# =============================================================================

class ScoringStrategy(ABC):
    """Abstract strategy for scoring."""

    @abstractmethod
    def score(self, data: Dict[str, Any]) -> float:
        """Calculate score for data."""
        pass


class LengthScorer(ScoringStrategy):
    """Score based on text length."""

    def score(self, data: Dict[str, Any]) -> float:
        text = data.get("text", "")
        return min(1.0, len(text) / 100)


class KeywordScorer(ScoringStrategy):
    """Score based on keyword presence."""

    def __init__(self, keywords: List[str]):
        self.keywords = keywords

    def score(self, data: Dict[str, Any]) -> float:
        text = data.get("text", "").lower()
        matches = sum(1 for kw in self.keywords if kw.lower() in text)
        return min(1.0, matches / len(self.keywords)) if self.keywords else 0.0


class CompositeScorer(ScoringStrategy):
    """Combine multiple scoring strategies."""

    def __init__(self, strategies: List[tuple]):
        """Initialize with (strategy, weight) pairs."""
        self.strategies = strategies

    def score(self, data: Dict[str, Any]) -> float:
        total = 0.0
        weight_sum = 0.0

        for strategy, weight in self.strategies:
            total += strategy.score(data) * weight
            weight_sum += weight

        return total / weight_sum if weight_sum > 0 else 0.0


def demonstrate_strategy_pattern() -> None:
    """Show strategy pattern."""
    print("Part 3: Strategy Pattern")
    print("-" * 50)

    data = {"text": "Machine learning and artificial intelligence are transforming technology."}

    # Individual strategies
    length_scorer = LengthScorer()
    keyword_scorer = KeywordScorer(["machine", "learning", "AI", "technology"])

    print(f"Length score: {length_scorer.score(data):.2f}")
    print(f"Keyword score: {keyword_scorer.score(data):.2f}")

    # Composite strategy
    composite = CompositeScorer([
        (length_scorer, 0.3),
        (keyword_scorer, 0.7),
    ])
    print(f"Composite score: {composite.score(data):.2f}")
    print()


# =============================================================================
# Part 4: Builder Pattern for Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration built by the builder."""

    steps: List[str] = field(default_factory=list)
    parallel: bool = False
    max_retries: int = 3
    timeout: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineBuilder:
    """Builder for creating pipeline configurations."""

    def __init__(self):
        self._config = PipelineConfig()

    def add_step(self, step: str) -> "PipelineBuilder":
        """Add a processing step."""
        self._config.steps.append(step)
        return self

    def parallel(self, enabled: bool = True) -> "PipelineBuilder":
        """Enable parallel execution."""
        self._config.parallel = enabled
        return self

    def retries(self, count: int) -> "PipelineBuilder":
        """Set retry count."""
        self._config.max_retries = count
        return self

    def timeout(self, seconds: float) -> "PipelineBuilder":
        """Set timeout in seconds."""
        self._config.timeout = seconds
        return self

    def metadata(self, **kwargs: Any) -> "PipelineBuilder":
        """Add metadata."""
        self._config.metadata.update(kwargs)
        return self

    def build(self) -> PipelineConfig:
        """Build the configuration."""
        return self._config


def demonstrate_builder_pattern() -> None:
    """Show builder pattern."""
    print("Part 4: Builder Pattern")
    print("-" * 50)

    # Fluent builder
    config = (
        PipelineBuilder()
        .add_step("preprocess")
        .add_step("analyze")
        .add_step("summarize")
        .parallel(True)
        .retries(5)
        .timeout(60.0)
        .metadata(version="1.0", author="example")
        .build()
    )

    print(f"Steps: {config.steps}")
    print(f"Parallel: {config.parallel}")
    print(f"Max retries: {config.max_retries}")
    print(f"Timeout: {config.timeout}s")
    print(f"Metadata: {config.metadata}")
    print()


# =============================================================================
# Part 5: Decorator Stacking
# =============================================================================

def with_logging(func: Callable) -> Callable:
    """Decorator that adds logging."""
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"  [LOG] Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"  [LOG] Completed {func.__name__}")
        return result
    wrapper.__name__ = func.__name__
    return wrapper


def with_timing(func: Callable) -> Callable:
    """Decorator that adds timing."""
    import time

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"  [TIME] {func.__name__}: {elapsed * 1000:.2f}ms")
        return result
    wrapper.__name__ = func.__name__
    return wrapper


def with_retry(max_attempts: int = 3) -> Callable:
    """Decorator factory for retry logic."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt + 1 == max_attempts:
                        raise
                    print(f"  [RETRY] Attempt {attempt + 1} failed: {e}")
            return None
        wrapper.__name__ = func.__name__
        return wrapper
    return decorator


def demonstrate_decorator_stacking() -> None:
    """Show decorator stacking."""
    print("Part 5: Decorator Stacking")
    print("-" * 50)

    @with_logging
    @with_timing
    @op
    def process_data(x: int) -> int:
        return x * 2

    result = process_data(21)
    print(f"Result: {result}")
    print()


# =============================================================================
# Part 6: Type-Safe Registry
# =============================================================================

class TypeRegistry:
    """Type-safe registry with generics simulation."""

    def __init__(self):
        self._registry: Dict[str, tuple] = {}

    def register(self, name: str, handler: Callable, input_type: Type, output_type: Type) -> None:
        """Register a typed handler."""
        self._registry[name] = (handler, input_type, output_type)

    def get(self, name: str) -> Optional[Callable]:
        """Get a handler by name."""
        entry = self._registry.get(name)
        return entry[0] if entry else None

    def call_typed(self, name: str, value: Any) -> Any:
        """Call handler with type checking."""
        entry = self._registry.get(name)
        if not entry:
            raise KeyError(f"Unknown handler: {name}")

        handler, input_type, output_type = entry

        if not isinstance(value, input_type):
            raise TypeError(f"Expected {input_type.__name__}, got {type(value).__name__}")

        result = handler(value)

        if not isinstance(result, output_type):
            raise TypeError(f"Handler returned {type(result).__name__}, expected {output_type.__name__}")

        return result

    def list_handlers(self) -> List[Dict[str, str]]:
        """List all registered handlers with types."""
        return [
            {
                "name": name,
                "input": entry[1].__name__,
                "output": entry[2].__name__,
            }
            for name, entry in self._registry.items()
        ]


def demonstrate_type_registry() -> None:
    """Show type-safe registry."""
    print("Part 6: Type-Safe Registry")
    print("-" * 50)

    registry = TypeRegistry()

    # Register typed handlers
    registry.register("double", lambda x: x * 2, int, int)
    registry.register("stringify", lambda x: str(x), int, str)
    registry.register("length", lambda x: len(x), str, int)

    # List handlers
    print("Registered handlers:")
    for handler in registry.list_handlers():
        print(f"  {handler['name']}: {handler['input']} -> {handler['output']}")
    print()

    # Call typed
    result = registry.call_typed("double", 21)
    print(f"double(21) = {result}")

    result = registry.call_typed("stringify", 42)
    print(f"stringify(42) = '{result}'")
    print()


def main() -> None:
    """Demonstrate advanced techniques."""
    print("Advanced Techniques")
    print("=" * 50)
    print()

    demonstrate_dynamic_composition()
    demonstrate_plugin_architecture()
    demonstrate_strategy_pattern()
    demonstrate_builder_pattern()
    demonstrate_decorator_stacking()
    demonstrate_type_registry()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Use factories for dynamic operator creation")
    print("2. Plugin architectures enable extensibility")
    print("3. Strategy pattern allows swappable algorithms")
    print("4. Builder pattern creates readable configurations")
    print("5. Decorator stacking composes cross-cutting concerns")
    print("6. Type registries provide runtime type safety")
    print()
    print("Next: See jax_xcs_integration.py for XCS/JAX patterns")


if __name__ == "__main__":
    main()
