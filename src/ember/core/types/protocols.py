"""
Core protocols for the Ember type system.

This module defines protocol classes that establish interfaces for
various components in the Ember system, enabling structural typing
and better interoperability between components.
"""

from typing import Any, Dict, Protocol, runtime_checkable, Type, Optional


class TypeInfo:
    """
    Metadata container for type information.

    This class provides a simple, lightweight container for type metadata
    that can be used at runtime for type inspection and validation.
    """

    def __init__(
        self,
        origin_type: Type,
        type_args: Optional[tuple] = None,
        is_optional: bool = False,
        is_container: bool = False,
    ) -> None:
        """
        Initialize TypeInfo with basic type metadata.

        Args:
            origin_type: The base type (e.g., dict, list, str)
            type_args: Tuple of type arguments for generic types
            is_optional: Whether the type is Optional[...]
            is_container: Whether the type is a container (list, dict, etc.)
        """
        self.origin_type = origin_type
        self.type_args = type_args or ()
        self.is_optional = is_optional
        self.is_container = is_container


@runtime_checkable
class EmberTyped(Protocol):
    """
    Protocol for objects that participate in Ember's type system.

    This is the most basic protocol that all typed objects in the Ember
    system should implement. It provides methods for type introspection.
    """

    def get_type_info(self) -> TypeInfo:
        """
        Return type metadata for this object.

        Returns:
            TypeInfo: Metadata about this object's type
        """
        ...


@runtime_checkable
class EmberSerializable(Protocol):
    """
    Protocol for objects that can be serialized to/from primitive types.

    This protocol defines the interface for objects that need to be
    serialized to and from various formats (dict, JSON, etc.)
    """

    def as_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        ...

    def as_json(self) -> str:
        """
        Convert to a JSON string.

        Returns:
            str: JSON string representation
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Any:
        """
        Create an instance from a dictionary.

        Args:
            data: Dictionary with serialized data

        Returns:
            An instance of this class
        """
        ...
