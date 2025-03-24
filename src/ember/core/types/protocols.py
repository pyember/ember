"""
Core protocols for the Ember type system.

Defines the fundamental interface protocols that establish consistent contracts
across the Ember framework. These protocols follow a focused, SOLID design:
- Each protocol has a single responsibility
- Interfaces are minimal and focused
- Dependencies are inverted through protocol abstractions
- Clear separation between type information and serialization concerns
"""

from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)


class TypeInfo:
    """
    Metadata container for runtime type information.

    Contains essential type metadata for runtime type checking, validation,
    and introspection, enabling dynamic behavior based on types.
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


T = TypeVar("T")


@runtime_checkable
class TypedProtocol(Protocol):
    """
    Protocol for objects with inspectable type information.

    Provides basic type introspection capabilities, separated from serialization
    concerns to follow Interface Segregation Principle.
    """

    def get_type_info(self) -> TypeInfo:
        """
        Return type metadata for this object.

        Returns:
            TypeInfo: Metadata about this object's type
        """
        ...


@runtime_checkable
class Serializable(Protocol, Generic[T]):
    """
    Protocol for objects with consistent serialization and deserialization.

    Provides a clear interface for converting objects to and from different
    serialization formats. Generic type parameter ensures type-safe deserialization.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary representation
        """
        ...

    def to_json(self) -> str:
        """
        Convert to a JSON string.

        Returns:
            str: JSON string representation
        """
        ...

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create an instance from a dictionary.

        Args:
            data: Dictionary with serialized data

        Returns:
            An instance of this class with proper type
        """
        ...

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """
        Create an instance from a JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            An instance of this class with proper type
        """
        ...


# Legacy aliases for backward compatibility
EmberTyped = TypedProtocol
EmberSerializable = Serializable
