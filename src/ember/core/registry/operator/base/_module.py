"""
Core EmberModule Abstraction.

This module provides the base class ``EmberModule``, a frozen dataclass that is
automatically registered with the transformation tree system.
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import field
from typing import Any, Callable, Dict, List, Type, Optional

from src.ember.xcs.utils.tree_util import register_tree, tree_flatten


def static_field(**kwargs: Any) -> dataclasses.Field:
    """Create a dataclass field marked as static.

    Static fields are excluded from tree transformations.

    Args:
        **kwargs: Additional keyword arguments passed to dataclasses.field.

    Returns:
        dataclasses.Field: A field configured as static.
    """
    return field(metadata={"static": True}, **kwargs)


def ember_field(
    *,
    converter: Optional[Callable[[Any], Any]] = None,
    static: bool = False,
    **kwargs: Any,
) -> dataclasses.Field:
    """Create a dataclass field with Ember-specific functionality.

    Supports defining converters and marking fields as static.

    Args:
        converter (Optional[Callable[[Any], Any]]): Function to convert the field's
            value during initialization.
        static (bool): If True, marks the field as static (excluded from tree transformations).
        **kwargs: Additional keyword arguments passed to dataclasses.field.

    Returns:
        dataclasses.Field: A configured dataclass field.
    """
    metadata: Dict[str, Any] = kwargs.pop("metadata", {})
    if converter is not None:
        metadata["converter"] = converter
    if static:
        metadata["static"] = True
    return field(metadata=metadata, **kwargs)


def _make_initable_wrapper(cls: Type[Any]) -> Type[Any]:
    """Create a temporary mutable (initable) wrapper for a frozen class.

    This wrapper allows mutations during initialization (__init__ and __post_init__).
    Once initialization completes, the instance's class is reset to the original frozen class.

    Args:
        cls (Type[Any]): The original frozen class.

    Returns:
        Type[Any]: A mutable subclass of cls.
    """

    class Initable(cls):  # type: ignore
        def __setattr__(self, name: str, value: Any) -> None:
            """Allow attribute mutation during initialization."""
            object.__setattr__(self, name, value)

    Initable.__name__ = cls.__name__
    Initable.__qualname__ = cls.__qualname__
    Initable.__module__ = cls.__module__
    return Initable


def _flatten_ember_module(instance: Any) -> tuple[List[Any], Dict[str, Any]]:
    """Flatten an EmberModule instance into dynamic and static fields.

    Dynamic fields participate in tree transformations while static fields remain fixed.

    Args:
        instance (Any): The EmberModule instance to flatten.

    Returns:
        tuple: A tuple of a list of dynamic field values and a dict of static field values.
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
    """Reconstruct an EmberModule instance from flattened components.

    Children values are assigned to dynamic field names and auxiliary static data is added.

    Args:
        cls (Type[Any]): The EmberModule class to instantiate.
        aux (Dict[str, Any]): Auxiliary mapping of static field names to values.
        children (List[Any]): List of dynamic field values.

    Returns:
        Any: A reconstructed instance of cls.
    """
    field_names: List[str] = [
        field_info.name
        for field_info in dataclasses.fields(cls)
        if not field_info.metadata.get("static", False)
    ]
    kwargs: Dict[str, Any] = dict(zip(field_names, children))
    kwargs.update(aux)
    return cls(**kwargs)


class EmberModuleMeta(abc.ABCMeta):
    """Metaclass for EmberModule.

    Ensures that every EmberModule subclass is a frozen dataclass and registers it with
    the transformation tree system. It also implements a mutable initialization phase by using
    a temporary wrapper during __init__ and __post_init__.
    """

    def __new__(
        cls,
        name: str,
        bases: tuple[Type[Any], ...],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ) -> type:
        """Create a new EmberModule subclass.

        Args:
            name (str): The name of the class.
            bases (tuple[Type[Any], ...]): Base classes.
            namespace (Dict[str, Any]): Class attributes.
            **kwargs: Additional keyword arguments.

        Returns:
            type: The newly created EmberModule subclass.
        """
        new_class: type = super().__new__(cls, name, bases, namespace, **kwargs)

        if not dataclasses.is_dataclass(new_class):
            new_class = dataclasses.dataclass(frozen=True, init=True)(new_class)

        register_tree(
            cls=new_class,
            flatten_func=lambda inst: _flatten_ember_module(inst),
            unflatten_func=lambda *, aux, children: _unflatten_ember_module(
                cls=new_class, aux=aux, children=children
            ),
        )
        return new_class

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Override instantiation to allow a temporary mutable initialization phase.

        A temporary mutable wrapper allows __init__ and __post_init__ to assign fields.
        After initialization, the instance is frozen again.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            Any: A fully initialized and frozen EmberModule instance.
        """
        mutable_cls = _make_initable_wrapper(cls)
        instance = super(EmberModuleMeta, mutable_cls).__call__(*args, **kwargs)
        object.__setattr__(instance, "__class__", cls)
        return instance


class EmberModule(metaclass=EmberModuleMeta):
    """Base class for Ember modules.

    Subclass EmberModule to create immutable, strongly typed modules that are automatically
    registered with the transformation tree system.
    """

    def _init_field(self, name: str, value: Any) -> None:
        """Helper method for mutating fields during initialization.

        This should be used in __post_init__ to set computed or default fields.

        Args:
            name (str): The name of the field.
            value (Any): The value to assign to the field.
        """
        object.__setattr__(self, name, value)

    def __hash__(self) -> int:
        """Compute a hash based on the dynamic fields.

        Returns:
            int: The computed hash value.
        """
        dynamic, _ = _flatten_ember_module(self)
        return hash(tuple(dynamic))

    def __eq__(self, other: Any) -> bool:
        """Check equality based on the flattened module data.

        Args:
            other (Any): The object to compare against.

        Returns:
            bool: True if both instances have equal flattened data, False otherwise.
        """
        if not isinstance(other, EmberModule):
            return False
        return tree_flatten(tree=self) == tree_flatten(tree=other)

    def __repr__(self) -> str:
        """Return a string representation of the EmberModule.

        Returns:
            str: A string representation of the instance.
        """
        return f"{self.__class__.__name__}({dataclasses.asdict(self)})"


class BoundMethod(EmberModule):
    """A bound method that also participates in the transformation tree system."""

    __func__: Callable[..., Any] = ember_field(static=True)
    __self__: EmberModule

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the bound method with the provided arguments.

        Args:
            *args (Any): Positional arguments for the underlying function.
            **kwargs (Any): Keyword arguments for the underlying function.

        Returns:
            Any: The result of calling the bound function.
        """
        return self.__func__(self.__self__, *args, **kwargs)

    @property
    def __wrapped__(self) -> Callable[..., Any]:
        """Retrieve the underlying function bound to __self__.

        Returns:
            Callable[..., Any]: The underlying bound function.
        """
        return self.__func__.__get__(self.__self__, type(self.__self__))
