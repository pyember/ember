"""
Core EmberModule abstraction for immutable tree-transformable modules.

This module establishes the foundation for Ember's immutable, tree-transformable
module system. The base classes enable creation of modules that:

  1. Are immutable (frozen) dataclasses with controlled initialization.
  2. Automatically register with the transformation tree system.
  3. Support static fields that are excluded from tree transformations.
  4. Support custom converters for field initialization.
  5. Include thread-safe flattening/unflattening for tree transformations.

The EmberModule system is optimized for JAX-like transformations such as just-in-time
(JIT) compilation, maps, and future gradient computation while preserving
compatibility with a native Pythonic, object-oriented programming model.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import logging
import threading
from dataclasses import Field, field
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    final,
)

from ember.core.registry.operator.exceptions import (
    BoundMethodNotInitializedError,
    FlattenError,
)
from ember.xcs.utils.tree_util import register_tree, tree_flatten

T = TypeVar("T")
EmberT = TypeVar("EmberT", bound="EmberModule")
FuncT = TypeVar("FuncT", bound=Callable[..., Any])

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_LOGGER: logging.Logger = logging.getLogger(__name__)


class ModuleCache:
    """Thread-safe, memory-efficient LRU cache for EmberModule flattening.

    This class provides a thread-isolated caching system optimized for
    transformation operations on EmberModule instances. It uses thread-local
    storage with identity-based keys (object IDs) to eliminate dependencies on
    hash or equality methods, preventing recursion issues when flattening objects
    that might themselves use the cache.

    The implementation offers:
    1. Thread isolation: Each thread has its own independent cache
    2. Memory safety: Tracking object lifetimes without circular dependencies
    3. Bounded growth: LRU eviction prevents unbounded memory consumption
    4. High performance: O(1) lookups and updates for common operations

    Technical details:
    - Uses thread.local() for thread isolation without lock contention
    - Implements the LRU algorithm using OrderedDict for O(1) reordering
    - Uses object identity (id) as cache keys to avoid hash method dependencies
    - Handles lifetime management through explicit API rather than weak references

    Thread-safety guarantees:
    - Thread-safe for all operations without locks (thread isolation)
    - No race conditions for cache updates within a thread
    - No shared state between threads, eliminating concurrency issues

    This cache is specifically optimized for transformation systems that apply
    the same operations to the same modules without creating circular dependencies.
    """

    DEFAULT_MAX_CACHE_SIZE: Final[int] = 1000
    """Default maximum number of entries to store in each thread's cache."""

    def __init__(self, max_cache_size: Optional[int] = None) -> None:
        """Initialize a new module cache with specified capacity.

        Args:
            max_cache_size: Maximum number of entries to store in each thread's cache.
                If None, uses DEFAULT_MAX_CACHE_SIZE (1000).
        """
        self._thread_local = threading.local()
        self._max_cache_size: int = max_cache_size or self.DEFAULT_MAX_CACHE_SIZE

    def _get_cache(
        self,
    ) -> Tuple[
        Dict[int, Tuple[List[object], Dict[str, object]]], collections.OrderedDict
    ]:
        """Initializes and returns the thread-local cache structures.

        Creates the thread-local cache structures if they don't exist yet.
        Uses a regular dictionary with object IDs as keys for the main cache to avoid
        relying on object hash methods, and OrderedDict to track LRU order.

        Returns:
            A tuple containing (flatten_cache, lru_order) for the current thread.
        """
        if not hasattr(self._thread_local, "flatten_cache"):
            self._thread_local.flatten_cache = {}
            self._thread_local.lru_order = collections.OrderedDict()

        return (self._thread_local.flatten_cache, self._thread_local.lru_order)

    def get(self, instance: object) -> Optional[Tuple[List[object], Dict[str, object]]]:
        """Retrieves a cached flattened representation if available.

        Implements the read path of the cache with LRU update. If the requested
        instance is in the cache, it's moved to the most-recently-used position
        in the LRU order before returning the cached value.

        Args:
            instance: The EmberModule instance to look up.

        Returns:
            The cached flattened representation as (dynamic_fields, static_fields),
            or None if the instance is not in the cache.
        """
        cache, lru_order = self._get_cache()
        instance_id = id(instance)
        result = cache.get(instance_id)

        if result is not None:
            # Update LRU order - move to end (most recently used position)
            lru_order.move_to_end(instance_id)

        return result

    def set(
        self, instance: object, value: Tuple[List[object], Dict[str, object]]
    ) -> None:
        """Stores a flattened representation in the cache.

        Implements the write path of the cache with LRU management. If adding this
        entry would exceed the maximum cache size, the least recently used entry
        is evicted before adding the new entry.

        Args:
            instance: The EmberModule instance to use as the cache key.
            value: The flattened representation to cache as
                (dynamic_fields, static_fields).
        """
        cache, lru_order = self._get_cache()
        instance_id = id(instance)

        if instance_id in cache:
            # Update existing entry's LRU position
            lru_order.move_to_end(instance_id)
        else:
            # Adding new entry - check if we need to evict
            if len(cache) >= self._max_cache_size:
                # Evict least recently used item (first item in OrderedDict)
                oldest_id, _ = lru_order.popitem(last=False)
                cache.pop(oldest_id, None)

            # Add to LRU tracking (value is irrelevant, only keys matter)
            lru_order[instance_id] = None

        # Store in main cache
        cache[instance_id] = value

    def clear(self, instance: Optional[object] = None) -> None:
        """Clears entries from the cache.

        Selectively clears either a specific entry or the entire cache for
        the current thread. This provides explicit control over cache contents
        for advanced memory management scenarios.

        Args:
            instance: If provided, only this instance's entry is cleared.
                     If None, the entire cache for the current thread is cleared.
        """
        cache, lru_order = self._get_cache()

        if instance is not None:
            # Clear specific entry
            instance_id = id(instance)
            cache.pop(instance_id, None)
            lru_order.pop(instance_id, None)
        else:
            # Clear entire thread-local cache
            cache.clear()
            lru_order.clear()

    def size(self) -> int:
        """Returns the current size of the thread-local cache.

        Provides visibility into the current cache utilization for the calling thread.
        This is useful for diagnostics and monitoring memory usage.

        Returns:
            The number of entries in the current thread's cache.
        """
        cache, _ = self._get_cache()
        return len(cache)

    def iter_entries(
        self,
    ) -> Iterator[Tuple[int, Tuple[List[object], Dict[str, object]]]]:
        """Iterates over all cache entries in LRU order (least recently used first).

        This method is primarily intended for diagnostics and debugging.

        Returns:
            An iterator yielding (instance_id, cached_value) pairs in LRU order.
        """
        cache, lru_order = self._get_cache()
        for instance_id in lru_order:
            if instance_id in cache:
                yield (instance_id, cache[instance_id])


# Global instance of the module cache
_module_cache = ModuleCache()


def static_field(*, default: object = dataclasses.MISSING, **kwargs: Any) -> Field:
    """Factory function that creates a dataclass field marked as static.

    Creates a dataclass field marked as static and excluded from tree transformations.

    Static fields are appropriate for configuration parameters and hyperparameters
    that should not be transformed by operations like JIT compilation or function mapping.
    These fields are preserved during tree transformation but are not passed through
    transformation functions.

    Example use cases:
    - Configuration dictionaries
    - Hyperparameters like learning rates or thresholds
    - Model metadata that shouldn't change during transformations
    - References to resources like tokenizers or vocabularies

    Example:
    ```python
    @dataclass
    class MyOperator(EmberModule):
        # Static field with default
        config: Dict[str, Any] = static_field(default_factory=dict)

        # Static field with specific default
        threshold: float = static_field(default=0.5)
    ```

    Args:
        default: Default value for the field if not provided during initialization.
        **kwargs: Additional keyword arguments for dataclasses.field.

    Returns:
        Field: A dataclass field configured as static.
    """
    metadata: Dict[str, Any] = kwargs.pop("metadata", {})
    metadata["static"] = True
    return field(default=default, metadata=metadata, **kwargs)


def ember_field(
    *,
    converter: Optional[Callable[[object], object]] = None,
    static: bool = False,
    default: object = dataclasses.MISSING,
    default_factory: Any = dataclasses.MISSING,
    init: bool = True,
    **kwargs: Any,
) -> Field:
    """Factory function that creates a dataclass field with Ember-specific functionality.

    Creates a dataclass field with Ember-specific functionality.

    This field provides enhanced capabilities for EmberModule fields, including:

    1. Value conversion: Apply transformations to field values during initialization
    2. Static marking: Control whether fields participate in tree transformations
    3. Default values: Standard dataclass field defaults and factories
    4. Initialization control: Configure whether fields appear in __init__ signatures

    The field converters run after instance initialization but before the instance
    is frozen, allowing for value validation, normalization, or type conversion.

    Example:
    ```python
    @dataclass
    class MyOperator(EmberModule):
        # Dynamic field (included in transformations)
        params: np.ndarray = ember_field()

        # Static field with converter
        config: Dict[str, Any] = ember_field(
            static=True,
            default_factory=dict,
            converter=lambda d: {**{'scale': 1.0}, **d}
        )

        # Computed field not included in initialization
        stats: Dict[str, float] = ember_field(init=False, default_factory=dict)

        def __post_init__(self):
            # Computed field example
            self._init_field(field_name="stats", value={"sum": sum(self.params)})
    ```

    Args:
        converter: Optional function to convert the field's value during initialization.
        static: If True, marks the field as static (excluded from transformations).
        default: Default value for the field if not provided during initialization.
        default_factory: Factory function to create the default value.
        init: Whether the field should be included in the __init__ parameters.
        **kwargs: Additional keyword arguments passed to dataclasses.field.

    Returns:
        Field: A dataclass field configured with Ember-specific settings.
    """
    metadata: Dict[str, Any] = kwargs.pop("metadata", {})
    if converter is not None:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True

    # pylint: disable=invalid-field-call
    # These calls to field() can appear invalid to pylint because they're not directly within
    # dataclass, but they're returned and used within dataclasses later
    if default is not dataclasses.MISSING:
        return field(default=default, metadata=metadata, init=init, **kwargs)

    if default_factory is not dataclasses.MISSING:
        return field(
            default_factory=default_factory, metadata=metadata, init=init, **kwargs
        )

    return field(metadata=metadata, init=init, **kwargs)


def _make_initable_wrapper(cls: Type[T]) -> Type[T]:
    """Creates a temporary mutable wrapper for a frozen dataclass.

    This function solves a fundamental issue with frozen dataclasses: the inability to
    set attributes during initialization. It creates a subclass that temporarily overrides
    __setattr__ to allow mutation during initialization, enabling proper field setup in
    __init__ and __post_init__ methods.

    The approach preserves the benefits of immutability while eliminating the main
    drawback – the inability to easily initialize computed fields or perform validation
    during initialization.

    Technical implementation:
    - Creates a dynamic subclass with an overridden __setattr__ method
    - Preserves all class metadata (name, module, qualname) for proper reflection
    - The original __call__ method later swaps the instance back to the frozen class

    This design pattern is inspired by functional programming principles where
    initialization and mutation are strictly separated from the object's normal
    lifecycle.

    Args:
        cls: The original frozen dataclass to wrap.

    Returns:
        A temporary mutable subclass with identical type signatures but allowing
        mutation.
    """

    # pylint: disable=too-few-public-methods
    class Initable(cls):  # type: ignore
        """Temporary mutable wrapper class for initialization."""

        def __setattr__(self, name: str, value: Any) -> None:
            """Override to allow mutation during initialization phase.

            Args:
                name: The attribute name to set.
                value: The value to assign to the attribute.
            """
            object.__setattr__(self, name, value)

    # Copy class metadata to ensure proper reflection
    Initable.__name__ = cls.__name__
    Initable.__qualname__ = cls.__qualname__
    Initable.__module__ = cls.__module__
    return Initable


def _flatten_ember_module(
    instance: EmberModule,
    # max_cache_size is unused but kept for API compatibility
    max_cache_size: Optional[int] = None,  # pylint: disable=unused-argument
) -> Tuple[List[object], Dict[str, object]]:
    """Flattens an EmberModule instance into dynamic and static components.

    This function transforms an EmberModule instance into a representation suitable
    for tree-based transformations (like those in JAX), separating field values into
    dynamic fields (which participate in transformations) and static fields (which
    remain unchanged during transformations).

    The implementation uses thread-local caching to optimize performance for repeated
    flattening operations on the same instance, which is common in transformation
    systems. The cache is keyed by instance identity and uses weak references to
    prevent memory leaks.

    Performance characteristics:
    - First flatten: O(n) where n is the number of fields
    - Subsequent flattens of the same instance: O(1) due to caching
    - Memory usage: Bounded by the cache size limit

    Args:
        instance: The EmberModule instance to flatten.
        max_cache_size: Optional maximum size for the thread-local cache.
                     If None, uses the default cache size.

    Returns:
        A tuple containing:
        - List of dynamic field values (which participate in transformations)
        - Dictionary mapping static field names to their values (preserved)

    Raises:
        FlattenError: If field access fails or any other error occurs during flattening.
    """
    try:
        # Check cache first for O(1) fast path
        cached_result = _module_cache.get(instance)
        if cached_result is not None:
            return cached_result

        # Cache miss - perform actual flattening
        dynamic_fields: List[object] = []
        static_fields: Dict[str, object] = {}

        # Process each field according to its metadata
        for field_info in dataclasses.fields(instance):
            try:
                value: object = getattr(instance, field_info.name)
            except AttributeError as e:
                raise FlattenError(
                    f"Field '{field_info.name}' missing in instance"
                ) from e

            # Sort fields into dynamic (transformable) or static (preserved)
            if field_info.metadata.get("static", False):
                static_fields[field_info.name] = value
            else:
                dynamic_fields.append(value)

        # Cache result for future calls
        flattened = (dynamic_fields, static_fields)
        _module_cache.set(instance, flattened)
        return flattened

    except Exception as e:
        _LOGGER.exception("Error flattening EmberModule instance: %s", e)
        if isinstance(e, FlattenError):
            raise
        raise FlattenError(f"Failed to flatten EmberModule instance: {e}") from e


def _unflatten_ember_module(
    *, cls: Type[EmberT], aux: Dict[str, object], children: List[object]
) -> EmberT:
    """Reconstructs an EmberModule instance from flattened components.

    This function reconstructs an EmberModule instance from its flattened representation
    without invoking normal initialization. It's used by transformation systems to
    recreate modules after applying transformations to their dynamic fields.

    The reconstruction process:
    1. Identifies the names of dynamic fields from the class definition
    2. Validates that the number of dynamic fields matches the provided children
    3. Creates a new instance without calling __init__
    4. Sets dynamic fields from the transformed children values
    5. Sets static fields from the preserved auxiliary dictionary

    This low-level function is primarily used by tree transformation systems and
    should rarely be called directly.

    Implementation notes:
    - Bypasses normal initialization for efficiency and correctness after
      transformations
    - Preserves the exact order of dynamic fields for correct reconstruction
    - Uses direct object.__setattr__ to avoid triggering field conversion again
    - Performs validation to catch bugs in the transformation system

    Args:
        cls: The EmberModule class to instantiate.
        aux: Dictionary mapping static field names to their preserved values.
        children: List of dynamic field values in the same order as the class
            definition.

    Returns:
        An instance of the specified class reconstructed from the components.

    Raises:
        ValueError: If the number of dynamic fields does not match the number of
            children.
    """
    # Get the names of dynamic fields in order
    field_names: List[str] = [
        field_info.name
        for field_info in dataclasses.fields(cls)
        if not field_info.metadata.get("static", False)
    ]

    # Validate that we have the correct number of dynamic fields
    if len(field_names) != len(children):
        raise ValueError(
            f"Mismatch between number of dynamic fields ({len(field_names)}) and "
            f"provided children ({len(children)}) for class {cls.__name__}."
        )

    # Create a new instance directly, bypassing __init__
    instance = object.__new__(cls)

    # Set dynamic fields in the correct order
    for name, value in zip(field_names, children):
        object.__setattr__(instance, name, value)

    # Set static fields from auxiliary data
    for name, value in aux.items():
        object.__setattr__(instance, name, value)

    return instance


class EmberModuleMeta(abc.ABCMeta):
    """Metaclass for EmberModule providing automatic registration and initialization.

    This metaclass orchestrates several critical aspects of the EmberModule system:

    1. Dataclass automation: It ensures all EmberModule subclasses are properly
       decorated as frozen dataclasses, even if the developer forgets to apply
       the @dataclass decorator.

    2. Tree system registration: It automatically registers all EmberModule
       subclasses with the transformation tree system, enabling operations
       like JIT compilation to work transparently.

    3. Initialization framework: It manages a sophisticated initialization
       process that allows mutation during initialization but ensures
       immutability afterward, eliminating a common pain point with
       frozen dataclasses.

    4. Field conversion: It provides a post-initialization phase where field
       converters are applied, enabling validation and normalization.

    The metaclass uses a novel technique to temporarily make frozen classes
    mutable during initialization by swapping the class of the instance
    during the initialization phase.

    Advanced implementation notes:
    - Uses custom __call__ to override the standard instance creation process
    - Creates a temporary mutable subclass for the initialization phase
    - Handles field defaults and converters automatically
    - Provides detailed error messages for initialization failures
    """

    def __new__(
        mcs: Type[EmberModuleMeta],
        name: str,
        bases: Tuple[Type[Any], ...],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ) -> Type:
        """Creates a new EmberModule subclass with automatic registration.

        Args:
            mcs: The metaclass itself.
            name: The name of the class being created.
            bases: The base classes of the class being created.
            namespace: The attribute dictionary of the class being created.
            **kwargs: Additional keyword arguments for class creation.

        Returns:
            Type: The newly created class.
        """
        # Create the new class
        new_class: Type = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Ensure it's a frozen dataclass if not already
        if not dataclasses.is_dataclass(new_class):
            new_class = dataclasses.dataclass(frozen=True, init=True)(new_class)

        # Define wrapper functions for tree operations
        def flatten(inst: Any) -> Tuple[List[Any], Dict[str, Any]]:
            """Wrapper for flattening an EmberModule instance.

            Args:
                inst: The instance to flatten.

            Returns:
                Tuple[List[Any], Dict[str, Any]]: The flattened representation.
            """
            return _flatten_ember_module(instance=inst)

        def unflatten(*, aux: Dict[str, Any], children: List[Any]) -> Any:
            """Wrapper for unflattening into an EmberModule instance.

            Args:
                aux: Static field values.
                children: Dynamic field values.

            Returns:
                Any: The reconstructed instance.
            """
            return _unflatten_ember_module(cls=new_class, aux=aux, children=children)

        # Register with the tree system
        register_tree(
            cls=new_class,
            flatten_func=flatten,
            unflatten_func=unflatten,
        )
        return new_class

    # pylint: disable=too-many-locals,too-many-branches
    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Creates an instance with complete initialization, regardless of whether
        super().__init__() is called.

        This implementation entirely sidesteps the need for users to call
        super().__init__()
        by handling all field initialization before and after the custom __init__
        method.

        Args:
            cls: The class being instantiated.
            *args: Positional arguments for initialization.
            **kwargs: Keyword arguments for initialization.

        Returns:
            T: A fully initialized, immutable instance of the EmberModule subclass.
        """
        # Create a mutable wrapper for initialization
        mutable_cls: Type[T] = _make_initable_wrapper(cls)

        # Create an instance directly without calling __init__
        instance = object.__new__(mutable_cls)

        # First set defaults for all fields - this ensures all fields exist
        # even if a custom __init__ doesn't set them
        fields_dict = {f.name: f for f in dataclasses.fields(cls)}
        for field_name, field_def in fields_dict.items():
            if field_name not in kwargs:
                if field_def.default is not dataclasses.MISSING:
                    object.__setattr__(instance, field_name, field_def.default)
                elif field_def.default_factory is not dataclasses.MISSING:
                    object.__setattr__(
                        instance, field_name, field_def.default_factory()
                    )

        # Call the class's __init__ method if it exists
        has_custom_init = (
            hasattr(cls, "__init__") and cls.__init__ is not object.__init__
        )
        if has_custom_init:
            try:
                mutable_cls.__init__(instance, *args, **kwargs)
            except TypeError as e:
                raise TypeError(
                    f"Error initializing {cls.__name__}: {str(e)}\n"
                    f"Ensure __init__ accepts the correct parameters."
                ) from e
        else:
            # For classes without custom __init__, set fields from kwargs
            for field_name, value in kwargs.items():
                if field_name in fields_dict:
                    object.__setattr__(instance, field_name, value)

            # Call __post_init__ if it exists
            post_init = getattr(instance, "__post_init__", None)
            if callable(post_init):
                post_init()

        # Check for missing fields
        missing_fields = []
        for field_name, field_def in fields_dict.items():
            if field_name not in dir(instance):
                if (
                    field_def.default is dataclasses.MISSING
                    and field_def.default_factory is dataclasses.MISSING
                ):
                    missing_fields.append(field_name)

        if missing_fields:
            raise ValueError(
                f"The following fields were not initialized: {missing_fields}"
            )

        # Apply field converters
        for field_info in dataclasses.fields(cls):
            converter = field_info.metadata.get("converter", None)
            if converter is not None and hasattr(instance, field_info.name):
                current_value = getattr(instance, field_info.name)
                converted_value = converter(current_value)
                object.__setattr__(instance, field_info.name, converted_value)

        # Revert to the original frozen class
        object.__setattr__(instance, "__class__", cls)
        return instance


class EmberModule(metaclass=EmberModuleMeta):
    """Base class for Ember's immutable, transformable modules.

    EmberModule provides the foundation for building immutable, strongly-typed modules
    that can be transformed by tree-based operations like JIT compilation, gradients,
    and vectorization. It combines Python's dataclass system with custom initialization
    and tree traversal mechanisms to create a framework optimized for transformation-based
    computation.

    Architectural principles:

    1. Immutability: Instances are frozen after initialization to prevent unexpected
       state mutations, improving reasoning about code and enabling parallel execution.

    2. Tree transformability: Automatic registration with the tree transformation system
       enables higher-order operations like JIT compilation.

    3. Static/dynamic field separation: Fields can be marked as static (excluded from
       transformations) or dynamic (included in transformations).

    4. Field conversion: Values can be automatically converted during initialization
       through field converters.

    5. Memory efficiency: Thread-safe caching with lifecycle management prevents
       memory leaks in long-running applications.

    Usage:

    ```python
    from dataclasses import dataclass
    from ember.core.registry.operator.base._module import EmberModule, ember_field, static_field

    @dataclass
    class MyModule(EmberModule):
        # Dynamic field - included in transformations
        weights: torch.Tensor = ember_field()

        # Static field - excluded from transformations
        config: Dict[str, float] = static_field(default_factory=dict)

        def __post_init__(self):
            if 'scale' not in self.config:
                self._init_field(field_name='config',
                                value={**self.config, 'scale': 1.0})

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.weights * self.config['scale']
    ```

    Quick notes on performance character:

    - Initialization: O(n) where n is the number of fields
    - Flattening: O(1) with caching after first operation
    - Field access: O(1) native Python attribute access
    - Memory: Proportional to the number of fields + bounded cache size
    """

    def _init_field(self, *, field_name: str, value: object) -> None:
        """Sets a field during initialization phase.

        This method provides a safe way to initialize or update fields during the
        module's initialization phase (typically within __post_init__). It bypasses
        the frozen state of the dataclass, but is only effective during initialization.

        Implementation notes:
        - Uses object.__setattr__ to bypass dataclass immutability
        - Only works during initialization before the instance is frozen
        - For computed or derived fields that depend on other initialized values

        Args:
            field_name: Name of the field to initialize or update.
            value: Value to assign to the field.
        """
        object.__setattr__(self, field_name, value)

    # Thread-local storage for recursion detection
    _hash_recursion_detection = threading.local()

    def __hash__(self) -> int:
        """Computes hash value based on dynamic field values.

        Provides hash-based container compatibility (sets, dict keys) based on
        the module's dynamic field values. Static fields are excluded from hash
        computation as they don't affect the module's transformation identity.

        Includes recursion detection to prevent infinite loops when flattening
        involves hash-based operations.

        Returns:
            Hash value derived from dynamic fields only, or object ID if recursion detected.
        """
        # Initialize recursion set if needed
        if not hasattr(self._hash_recursion_detection, "objects"):
            self._hash_recursion_detection.objects = set()

        # Check if we're already computing hash for this object
        if id(self) in self._hash_recursion_detection.objects:
            # Recursion detected, use object identity as fallback
            return hash(id(self))

        # Set recursion flag and compute hash
        self._hash_recursion_detection.objects.add(id(self))
        try:
            dynamic_fields, _ = _flatten_ember_module(instance=self)
            return hash(tuple(dynamic_fields))
        finally:
            # Always clean up recursion flag
            self._hash_recursion_detection.objects.remove(id(self))

    def __eq__(self, other: object) -> bool:
        """Determines equality based on dynamic field values.

        Two EmberModule instances are considered equal if:
        1. They are instances of compatible types
        2. Their dynamic fields have equivalent values

        Static fields are intentionally excluded from equality checks to align
        with transformation semantics.

        Args:
            other: Object to compare with this instance.

        Returns:
            True if instances have equivalent dynamic fields; False otherwise.
        """
        if not isinstance(other, EmberModule):
            return False
        return tree_flatten(tree=self) == tree_flatten(tree=other)

    def __repr__(self) -> str:
        """Returns a debug-friendly string representation.

        Provides a human-readable representation showing the module's class name
        and all field values (both dynamic and static).

        Returns:
            String representation of format "ClassName(field1=value1, field2=value2, ...)"
        """
        return f"{self.__class__.__name__}({dataclasses.asdict(self)})"

    def __del__(self) -> None:
        """Cleans up resources when the instance is garbage collected.

        Automatically removes this instance from the flattening cache
        to prevent memory leaks in long-running applications.
        """
        try:
            _module_cache.clear(self)
        except Exception:  # pylint: disable=broad-except
            # Ignore exceptions during interpreter shutdown
            pass

    def __pytree_flatten__(self) -> Tuple[List[object], Dict[str, object]]:
        """Implements the JAX PyTree flattening protocol.

        This method enables EmberModule instances to participate in JAX-style
        transformations by exposing a compatible flattening interface. It separates
        the module's state into dynamic values (which participate in transformations)
        and static values (which remain unchanged).

        Returns:
            Tuple of (dynamic_field_values, static_field_mapping).
        """
        dynamic, aux = _flatten_ember_module(self)
        return dynamic, aux

    @classmethod
    def __pytree_unflatten__(
        cls: Type[EmberT], aux: Dict[str, object], children: List[object]
    ) -> EmberT:
        """Implements the JAX PyTree unflattening protocol.

        This method enables reconstruction of EmberModule instances from
        transformed values, supporting the full transformation lifecycle.
        It bypasses normal initialization to efficiently recreate instances
        from transformed components.

        Args:
            aux: Dictionary of static field values preserved during transformation.
            children: List of dynamic field values potentially modified by transformation.

        Returns:
            Reconstructed EmberModule instance with transformed values.
        """
        return _unflatten_ember_module(cls=cls, aux=aux, children=children)

    @classmethod
    def clear_cache(cls, instance: Optional[object] = None) -> None:
        """Explicitly clears entries from the module flattening cache.

        This method provides direct control over the caching system for
        advanced memory management scenarios. It can selectively clear
        a specific instance's cache entry or the entire thread-local cache.

        Args:
            instance: If provided, only this instance's cache entry is cleared.
                     If None, the entire cache for the current thread is cleared.
        """
        _module_cache.clear(instance)

    @classmethod
    def get_cache_size(cls) -> int:
        """Returns the current size of the thread-local flattening cache.

        This diagnostic method is useful for monitoring memory usage and
        debugging cache-related performance issues.

        Returns:
            Number of entries in the current thread's cache.
        """
        return _module_cache.size()

    @classmethod
    def iter_cache_entries(
        cls,
    ) -> Iterator[Tuple[object, Tuple[List[object], Dict[str, object]]]]:
        """Iterates over all cache entries in LRU order.

        This advanced debugging method provides visibility into the cache contents
        and ordering, which can be useful for diagnosing complex caching issues.

        Returns:
            Iterator yielding (instance, cached_value) pairs in LRU order.
        """
        return _module_cache.iter_entries()


@final
class BoundMethod(EmberModule, Generic[EmberT, FuncT]):
    """Encapsulates a function bound to an EmberModule instance for tree-compatible method calls.

    BoundMethod is a specialized EmberModule that wraps a method-instance binding in a way that
    can participate in tree transformations. It enables transformation systems to track and
    transform methods bound to module instances, which is essential for higher-order
    operations like JIT compilation of instance methods.

    This class implements the descriptor protocol to allow method-like invocation
    while preserving transformability. When called, it delegates to the bound function
    with the bound instance as the first argument, mimicking Python's standard method
    binding behavior.

    Design rationale:
    - Uses EmberModule as base class to participate in tree transformations
    - Implements __call__ for direct invocation (mimicking bound methods)
    - Provides __wrapped__ property for introspection and unwrapping
    - Maintains immutability guarantees through EmberModule inheritance

    Technical details:
    - __func__ is marked as static since it shouldn't be transformed
    - __self__ is dynamic to allow transformations of the bound instance
    - Thread-safety is inherited from emberModule

    Attributes:
        __func__: The function to be bound (stored as static field).
        __self__: The EmberModule instance to which the function is bound (dynamic field).
    """

    __func__: Callable[[EmberT], FuncT] = ember_field(static=True)
    __self__: EmberModule = ember_field()

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Invokes the bound method with the specified arguments.

        This implementation mimics Python's standard bound method behavior by
        inserting the bound instance (__self__) as the first argument to the
        function (__func__) and passing through all other arguments.

        Args:
            *args: Positional arguments to pass after the bound instance.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            Result of the bound function invocation.

        Raises:
            BoundMethodNotInitializedError: If either __func__ or __self__ is not initialized.
        """
        if self.__func__ is None or self.__self__ is None:
            raise BoundMethodNotInitializedError(
                "BoundMethod not fully initialized: missing __func__ or __self__."
            )
        return self.__func__(self.__self__, *args, **kwargs)

    @property
    def __wrapped__(self) -> Callable[..., object]:
        """Retrieves the standard Python bound method equivalent.

        This property provides access to the standard Python bound method
        equivalent of this BoundMethod instance, which is useful for introspection,
        debugging, and interoperability with libraries that expect standard
        bound methods.

        Implementation note: Uses Python's descriptor protocol __get__ to create
        a standard bound method from the function and instance.

        Returns:
            The equivalent standard Python bound method.
        """
        return self.__func__.__get__(self.__self__, type(self.__self__))

    def __repr__(self) -> str:
        """Returns a string representation of the BoundMethod.

        Provides a human-readable representation showing the bound function
        and instance, useful for debugging.

        Returns:
            String representation in the format "BoundMethod(func=<function>, self=<instance>)"
        """
        func_name = getattr(self.__func__, "__name__", str(self.__func__))
        self_repr = (
            getattr(self.__self__, "__class__", None).__name__
            if self.__self__
            else "None"
        )
        return f"BoundMethod(func={func_name}, self={self_repr})"
