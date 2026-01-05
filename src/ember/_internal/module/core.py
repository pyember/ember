"""Static-by-default module abstractions built on Equinox."""

from typing import Any, get_args, get_origin

import equinox as eqx
import jax


class EmberModuleMeta(type(eqx.Module)):
    """Metaclass that applies static-by-default field semantics.

    A field is marked static unless its annotation (recursively) includes a JAX array
    type or another Equinox module. This keeps orchestration metadata (model bindings,
    config objects, etc.) out of traced JAX leaves while preserving trainable arrays
    and nested modules as dynamic fields.

    To override, declare the field explicitly with ``eqx.field(static=...)``.
    """

    @classmethod
    def _annotation_contains_dynamic(cls, annotation: Any) -> bool:
        origin = get_origin(annotation)
        if origin is None:
            if not isinstance(annotation, type):
                return False
            if issubclass(annotation, (jax.Array, eqx.Module)):
                return True
            return False

        return any(cls._annotation_contains_dynamic(arg) for arg in get_args(annotation))

    def __new__(mcs, name, bases, namespace, **kwargs):
        annotations = namespace.get("__annotations__", {})
        missing = object()

        for field_name, field_type in annotations.items():
            # Skip if already has explicit field definition
            if field_name in namespace and hasattr(namespace.get(field_name), "metadata"):
                continue

            default_value = namespace.get(field_name, missing)

            is_definitely_static = not mcs._annotation_contains_dynamic(field_type)

            if is_definitely_static:
                if default_value is missing:
                    namespace[field_name] = eqx.field(static=True)
                else:
                    namespace[field_name] = eqx.field(static=True, default=default_value)

        return super().__new__(mcs, name, bases, namespace, **kwargs)

    def __instancecheck__(cls, instance):
        if getattr(instance, "_is_operator_proxy", False):
            OperatorType: type[object] | None = None
            try:
                from ember.operators.base import Operator as OperatorType
            except ImportError:  # pragma: no cover
                pass

            target = getattr(instance, "_instance", None)
            if target is not None:
                return super().__instancecheck__(target)
            if OperatorType is not None and issubclass(cls, OperatorType):
                return True
            return False
        return super().__instancecheck__(instance)


class Module(eqx.Module, metaclass=EmberModuleMeta):
    """Equinox module with static-by-default semantics.

    Non-JAX fields (strings, mappings, configuration objects) are marked static
    automatically while JAX arrays remain dynamic for transformation support.

    Examples:
        >>> class MyOperator(Module):
        ...     name: str
        ...     config: dict
        ...     weights: "jax.Array"  # doctest: +SKIP
    """

    def __init_subclass__(cls, **kwargs):
        """Enhance subclasses with helpful error messages."""
        super().__init_subclass__(**kwargs)

        if not getattr(cls, "__annotations__", {}):
            # No annotations - wrap __init__ to provide helpful error on attribute errors
            if hasattr(cls, "__init__"):
                original_init = cls.__init__

                def init_with_helpful_errors(self, *args, **kwargs):
                    try:
                        original_init(self, *args, **kwargs)
                    except AttributeError as e:
                        if "Cannot set attribute" in str(e):
                            raise AttributeError(
                                f"{str(e)}\n\n"
                                f"Hint: Ember Modules require fields to be declared at the "
                                f"class level.\n"
                                f"Add field annotations to your class:\n\n"
                                f"class {cls.__name__}(Module):\n"
                                f"    field_name: field_type  # Add this before __init__\n"
                                f"\n"
                                f"    def __init__(self, ...):\n"
                                f"        self.field_name = value  # Now this will work\n\n"
                                f"For JAX arrays, use 'jax.Array' as the type annotation.\n"
                                f"For static config dicts, use: "
                                f"config: dict = eqx.field(static=True)"
                            ) from e
                        raise

                cls.__init__ = init_with_helpful_errors


# Alias for field to avoid leaking equinox abstractions
field = eqx.field

__all__ = ["Module", "field"]
