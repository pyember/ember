"""
Core EmberModule abstraction for immutable tree-transformable modules.

This module provides the foundation undergirding Ember's immutable, tree-transformable
module system. The base classes enable creation of modules that:

1. Are immutable (frozen) dataclasses with controlled initialization
2. Automatically register with the transformation tree system
3. Support static fields (excluded from tree transformations)
4. Support custom converters for field initialization
5. Include thread-safe flattening/unflattening for tree transformations

The EmberModule system is optimized for JAX-esque transformations such as just-in-time 
compilation, maps, and future gradient computation while preserving alignment with 
Python's object-oriented programming model.
"""

from __future__ import annotations

import abc
import dataclasses
import threading
from dataclasses import field, Field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Generic,
)

from ember.xcs.utils.tree_util import register_tree, tree_flatten
from ember.core.registry.operator.exceptions import (
    BoundMethodNotInitializedError,
    FlattenError,
)

T = TypeVar("T")
EmberT = TypeVar("EmberT", bound="EmberModule")
FuncT = TypeVar("FuncT", bound=Callable[..., Any])

# Thread-local storage for caching flattened representations
_thread_local = threading.local()


def static_field(*, default: Any = dataclasses.MISSING, **kwargs: Any) -> Field:
    """Creates a dataclass field marked as static and excluded from tree transformations.

    Static fields are appropriate for configuration parameters and hyperparameters
    that should not be transformed by operations like gradients or JIT.

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
    converter: Optional[Callable[[Any], Any]] = None,
    static: bool = False,
    default: Any = dataclasses.MISSING,
    default_factory: Any = dataclasses.MISSING,
    init: bool = True,
    **kwargs: Any,
) -> Field:
    """Creates a dataclass field with Ember-specific functionality.

    This field supports an optional converter for value initialization and can be marked as
    static to exclude it from tree transformations.

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

    if default is not dataclasses.MISSING:
        return field(default=default, metadata=metadata, init=init, **kwargs)
    elif default_factory is not dataclasses.MISSING:
        return field(
            default_factory=default_factory, metadata=metadata, init=init, **kwargs
        )
    else:
        return field(metadata=metadata, init=init, **kwargs)


def _make_initable_wrapper(cls: Type[T]) -> Type[T]:
    """Creates a temporary mutable wrapper for a frozen class.

    This wrapper allows mutations during initialization (i.e. __init__ and __post_init__),
    after which the instance's class is reverted to the original frozen class.

    Args:
        cls: The original frozen class.

    Returns:
        Type[T]: A mutable subclass of the original class.
    """

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


def _flatten_ember_module(instance: EmberModule) -> Tuple[List[Any], Dict[str, Any]]:
    """Flattens an EmberModule instance into its dynamic and static components.

    Dynamic fields participate in tree transformations, while static fields remain fixed.
    Uses thread-local caching for performance in transformation-heavy workloads.

    Args:
        instance: The EmberModule instance to flatten.

    Returns:
        Tuple[List[Any], Dict[str, Any]]: A tuple containing:
          - List of dynamic field values (transformed by tree operations)
          - Dictionary mapping static field names to their values (preserved)

    Raises:
        FlattenError: If a required field is missing from the instance.
    """
    # Initialize thread-local cache if needed
    if not hasattr(_thread_local, "flatten_cache"):
        _thread_local.flatten_cache = {}
    cache = _thread_local.flatten_cache
    instance_id = id(instance)

    # Return cached result if available
    if instance_id in cache:
        return cache[instance_id]

    dynamic_fields: List[Any] = []
    static_fields: Dict[str, Any] = {}

    # Process each field according to its metadata
    for field_info in dataclasses.fields(instance):
        try:
            value: Any = getattr(instance, field_info.name)
        except AttributeError:
            raise FlattenError(f"Field '{field_info.name}' missing in instance")

        # Sort fields into dynamic (transformable) or static (preserved)
        if field_info.metadata.get("static", False):
            static_fields[field_info.name] = value
        else:
            dynamic_fields.append(value)

    # Cache and return the result
    flattened = (dynamic_fields, static_fields)
    cache[instance_id] = flattened
    return flattened


def _unflatten_ember_module(
    *, cls: Type[EmberT], aux: Dict[str, Any], children: List[Any]
) -> EmberT:
    """Reconstructs an EmberModule instance from flattened components.

    This bypasses normal initialization to create an instance directly from components.

    Args:
        cls: The EmberModule class to instantiate.
        aux: A dictionary of static field values.
        children: A list of dynamic field values.

    Returns:
        EmberT: An instance of cls reconstructed from the provided components.

    Raises:
        ValueError: If the number of dynamic fields does not match the number of children.
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
            f"provided children ({len(children)})."
        )

    # Create a new instance directly
    instance = object.__new__(cls)

    # Set dynamic fields
    for name, value in zip(field_names, children):
        object.__setattr__(instance, name, value)

    # Set static fields
    for name, value in aux.items():
        object.__setattr__(instance, name, value)

    return instance


class EmberModuleMeta(abc.ABCMeta):
    """Metaclass for EmberModule.

    This metaclass:
      1. Automatically decorates subclasses as frozen dataclasses.
      2. Registers them with the transformation tree system.
      3. Implements a temporary mutable initialization phase via a wrapper.
      4. Applies field converters during initialization.
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

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Creates an instance with complete initialization, regardless of whether super().__init__() is called.

        This implementation entirely sidesteps the need for users to call super().__init__()
        by handling all field initialization before and after the custom __init__ method.

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

    Subclass EmberModule to create immutable, strongly-typed modules that integrate with
    the transformation tree system (e.g., for XCS `jit` or `grad` operations).

    Key features:
    - Immutable by default (frozen dataclasses)
    - Automatic registration with tree transformation system
    - Support for static fields (excluded from transformations)
    - Support for field converters (executed during initialization)
    - Thread-safe caching for improved performance

    Field types:
    - Fields marked with `static_field` or `ember_field(static=True)` are excluded from
      tree transformations, suitable for hyperparameters or fixed configurations.
    - Dynamic fields (default `ember_field`) participate in transformations,
      such as model/router weights getting backpropped through.

    Performance notes:
    - Flattening/unflattening has complexity proportional to the number of fields.
      Use `static_field` for fields that don't require transformation to reduce overhead.
    - Thread-local caching speeds up repeated flatten calls within the same thread.
    """

    def _init_field(self, *, field_name: str, value: Any) -> None:
        """Sets a field during initialization.

        Intended to be used within __post_init__ for computed or derived fields.
        Only works during the initialization phase before the instance is frozen.

        Args:
            field_name: The name of the field to initialize.
            value: The value to be assigned to the field.
        """
        object.__setattr__(self, field_name, value)

    def __hash__(self) -> int:
        """Computes a hash based on the dynamic fields.

        Returns:
            int: The computed hash value based on dynamic fields.
        """
        dynamic_fields, _ = _flatten_ember_module(instance=self)
        return hash(tuple(dynamic_fields))

    def __eq__(self, other: Any) -> bool:
        """Determines equality based on flattened field representations.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if both instances have equivalent dynamic field values; otherwise, False.
        """
        if not isinstance(other, EmberModule):
            return False
        return tree_flatten(tree=self) == tree_flatten(tree=other)

    def __repr__(self) -> str:
        """Returns a string representation of the EmberModule instance.

        Returns:
            str: A string showing the class name and its field values.
        """
        return f"{self.__class__.__name__}({dataclasses.asdict(self)})"

    def __pytree_flatten__(self) -> Tuple[List[Any], Dict[str, Any]]:
        """Skeuomorphic flatten protocol for JAX and similar libraries.

        Returns:
            Tuple[List[Any], Dict[str, Any]]: A tuple of (dynamic_fields, static_fields).
        """
        dynamic, aux = _flatten_ember_module(self)
        return dynamic, aux

    @classmethod
    def __pytree_unflatten__(
        cls: Type[EmberT], aux: Dict[str, Any], children: List[Any]
    ) -> EmberT:
        """Skeuomorphic unflatten protocol for JAX and similar libraries.

        Args:
            aux: The static fields preserved during transformation.
            children: A list of dynamic field values.

        Returns:
            EmberT: An instance of the EmberModule reconstructed from flattened components.
        """
        return _unflatten_ember_module(cls=cls, aux=aux, children=children)


class BoundMethod(EmberModule, Generic[EmberT, FuncT]):
    """Encapsulates a function bound to an EmberModule instance.

    Enables method-like invocation while participating in the transformation tree system.
    This facilitates tracking and transformation of methods bound to module instances.

    Attributes:
        __func__: The function to be bound.
        __self__: The EmberModule instance to which the function is bound.
    """

    __func__: Callable[..., Any] = ember_field(static=True)
    __self__: EmberModule = ember_field()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invokes the bound method with the specified arguments.

        Args:
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the bound function invocation.

        Raises:
            BoundMethodNotInitializedError: If either __func__ or __self__ is not initialized.
        """
        if self.__func__ is None or self.__self__ is None:
            raise BoundMethodNotInitializedError(
                "Bound method is not fully initialized."
            )
        return self.__func__(self.__self__, *args, **kwargs)

    @property
    def __wrapped__(self) -> Callable[..., Any]:
        """Retrieves the underlying function bound to the instance.

        Returns:
            Callable[..., Any]: The original function bound to __self__.
        """
        return self.__func__.__get__(self.__self__, type(self.__self__))
