"""
Core EmberModule Abstraction.

This module provides the base class ``EmberModule``, a frozen dataclass that is
automatically registered with the transformation tree system.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import field, Field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from src.ember.xcs.utils.tree_util import register_tree, tree_flatten
from src.ember.core.registry.operator.exceptions import BoundMethodNotInitializedError


def static_field(**kwargs: Any) -> Field:
    """Creates a dataclass field marked as static.

    Static fields are excluded from tree transformations.

    Args:
        **kwargs: Additional keyword arguments for dataclasses.field.

    Returns:
        Field: A dataclass field configured as static.
    """
    return field(metadata={"static": True}, **kwargs)


def ember_field(
    *,
    converter: Optional[Callable[[Any], Any]] = None,
    static: bool = False,
    **kwargs: Any,
) -> Field:
    """Creates a dataclass field with Ember-specific functionality.

    This field supports an optional converter for value initialization and can be marked as
    static to exclude it from tree transformations.

    Args:
        converter (Optional[Callable[[Any], Any]]): Optional function to convert the field's value
            during initialization.
        static (bool): If True, marks the field as static. Defaults to False.
        **kwargs: Additional keyword arguments for dataclasses.field.

    Returns:
        Field: A dataclass field configured with Ember-specific settings.
    """
    metadata: Dict[str, Any] = kwargs.pop("metadata", {})
    if converter is not None:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True
    return field(metadata=metadata, **kwargs)


def _make_initable_wrapper(cls: Type[Any]) -> Type[Any]:
    """Creates a temporary mutable wrapper for a frozen class.

    This wrapper allows mutations during initialization (i.e., __init__ and __post_init__),
    after which the instance's class is reverted to the original frozen class.

    Args:
        cls (Type[Any]): The original frozen class.

    Returns:
        Type[Any]: A mutable subclass of the original class.
    """
    class Initable(cls):  # type: ignore
        def __setattr__(self, name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

    Initable.__name__ = cls.__name__
    Initable.__qualname__ = cls.__qualname__
    Initable.__module__ = cls.__module__
    return Initable


def _flatten_ember_module(instance: Any) -> Tuple[List[Any], Dict[str, Any]]:
    """Flattens an EmberModule instance into its dynamic and static components.

    Dynamic fields participate in tree transformations, while static fields remain fixed.

    Args:
        instance (Any): The EmberModule instance to flatten.

    Returns:
        Tuple[List[Any], Dict[str, Any]]: A tuple containing a list of dynamic field values and
        a dictionary mapping static field names to their values.
    """
    dynamic_fields: List[Any] = []
    static_fields: Dict[str, Any] = {}
    for field_info in dataclasses.fields(instance):
        value: Any = getattr(instance, field_info.name)
        if field_info.metadata.get("static", False):
            static_fields[field_info.name] = value
        else:
            dynamic_fields.append(value)
    return dynamic_fields, static_fields


def _unflatten_ember_module(
    *, cls: Type[Any], aux: Dict[str, Any], children: List[Any]
) -> Any:
    """Reconstructs an EmberModule instance from flattened components.

    Args:
        cls (Type[Any]): The EmberModule class to instantiate.
        aux (Dict[str, Any]): A dictionary of static field values.
        children (List[Any]): A list of dynamic field values.

    Returns:
        Any: An instance of cls reconstructed from the provided components.

    Raises:
        ValueError: If the number of dynamic fields does not match the number of children.
    """
    field_names: List[str] = [
        field_info.name
        for field_info in dataclasses.fields(cls)
        if not field_info.metadata.get("static", False)
    ]
    if len(field_names) != len(children):
        raise ValueError("Mismatch between number of dynamic fields and provided children.")
    init_kwargs: Dict[str, Any] = dict(zip(field_names, children))
    init_kwargs.update(aux)
    return cls(**init_kwargs)


class EmberModuleMeta(abc.ABCMeta):
    """Metaclass for EmberModule.

    Automatically decorates subclasses as frozen dataclasses and registers them with the
    transformation tree system. Implements a temporary mutable initialization phase via a wrapper.
    """

    def __new__(
        cls,
        name: str,
        bases: Tuple[Type[Any], ...],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ) -> type:
        new_class: type = super().__new__(cls, name, bases, namespace, **kwargs)
        if not dataclasses.is_dataclass(new_class):
            new_class = dataclasses.dataclass(frozen=True, init=True)(new_class)

        def flatten(inst: Any) -> Tuple[List[Any], Dict[str, Any]]:
            """Wrapper for flattening an EmberModule instance."""
            return _flatten_ember_module(instance=inst)

        def unflatten(*, aux: Dict[str, Any], children: List[Any]) -> Any:
            """Wrapper for unflattening into an EmberModule instance."""
            return _unflatten_ember_module(cls=new_class, aux=aux, children=children)

        register_tree(
            cls=new_class,
            flatten_func=flatten,
            unflatten_func=unflatten,
        )
        return new_class

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Overrides instantiation to allow temporary mutable initialization.

        A mutable wrapper is used for __init__ and __post_init__ to set fields, after which
        the instance is frozen by resetting its class.

        Args:
            *args: Positional arguments for the instance.
            **kwargs: Keyword arguments for the instance.

        Returns:
            Any: An instance of the EmberModule subclass.
        """
        mutable_cls: Type[Any] = _make_initable_wrapper(cls)
        instance: Any = super(EmberModuleMeta, mutable_cls).__call__(*args, **kwargs)
        object.__setattr__(instance, "__class__", cls)
        return instance


class EmberModule(metaclass=EmberModuleMeta):
    """Base class for Ember modules.

    Subclass EmberModule to create immutable, strongly-typed modules that participate in the
    transformation tree system.
    """

    def _init_field(self, name: str, value: Any) -> None:
        """Sets a field during initialization.

        Intended to be used within __post_init__ for computed or derived fields.

        Args:
            name (str): The name of the field.
            value (Any): The value to be assigned to the field.
        """
        object.__setattr__(self, name, value)

    def __hash__(self) -> int:
        """Computes a hash based on the dynamic fields.

        Returns:
            int: The computed hash value.
        """
        dynamic_fields, _ = _flatten_ember_module(instance=self)
        return hash(tuple(dynamic_fields))

    def __eq__(self, other: Any) -> bool:
        """Determines equality based on flattened field representations.

        Args:
            other (Any): The object to compare with.

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


class BoundMethod(EmberModule):
    """Encapsulates a function bound to an EmberModule instance, enabling method-like invocation.

    Participates in the transformation tree system.
    """

    __func__: Callable[..., Any] = ember_field(static=True)
    __self__: EmberModule

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
