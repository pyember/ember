"""Base operator class for Ember.

This module provides the foundational Operator class that serves as the basis
for all AI components in Ember. Operators enable composition, transformation,
and optimization of AI systems through a clean, unified interface.

Operators inherit from equinox.Module, providing automatic JAX pytree
compatibility. Static fields (config, model names) are compile-time constants;
dynamic fields (JAX arrays) participate in transformations like grad/vmap.

Example:
    >>> class Summarizer(Operator):
    ...     def __init__(self):
    ...         self.model = ember.model("gpt-4")
    ...
    ...     def forward(self, text: str) -> str:
    ...         return self.model(f"Summarize: {text}")
    ...
    >>> summarizer = Summarizer()
    >>> summary = summarizer("Long document text...")
"""

import inspect
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple, Type, cast, get_type_hints

import equinox as eqx

from ember._internal.module import Module


def _effective_spec(
    instance: "Operator", attr_name: str, cached_spec: Optional[Type[Any]]
) -> Optional[Type[Any]]:
    """Return the spec for ``attr_name``.

    Checks instance (non-None only), then class, then cached inference.
    """
    cls = type(instance)
    value = instance.__dict__.get(attr_name)
    if value is not None:
        return cast(Type[Any], value)

    if attr_name in cls.__dict__:
        return cast(Type[Any], cls.__dict__[attr_name])

    return cached_spec


class Operator(Module):
    """Base operator class for building composable AI systems.

    Operators are composable building blocks that can be chained, ensembled,
    and transformed with JAX (grad, vmap, jit). Subclasses implement forward().

    Attributes:
        input_spec: Optional Pydantic model for input validation.
        output_spec: Optional Pydantic model for output validation.

    Examples:
        Basic operator:

        >>> class Classifier(Operator):
        ...     def __init__(self, model_name: str = "gpt-4"):
        ...         self.model = ember.model(model_name)
        ...
        ...     def forward(self, text: str) -> str:
        ...         return self.model(f"Classify: {text}").text

        With input/output validation:

        >>> class ValidatedClassifier(Operator):
        ...     input_spec = TextInput    # Pydantic model
        ...     output_spec = SentimentOutput
        ...
        ...     def forward(self, input: TextInput) -> SentimentOutput:
        ...         response = self.model(f"Analyze: {input.text}")
        ...         return SentimentOutput(label="positive", confidence=0.95)
        >>>
        >>> classifier({"text": "Great!"})  # Dict auto-validated to TextInput

        With learnable JAX parameters:

        >>> class LearnableClassifier(Operator):
        ...     def __init__(self, num_classes: int, key: jax.Array):
        ...         self.model = ember.model("gpt-4")
        ...         self.class_weights = jax.random.normal(key, (num_classes,))
        ...
        ...     def forward(self, text: str) -> int:
        ...         scores = self.compute_scores(self.model(f"Classify: {text}"))
        ...         return jnp.argmax(scores * self.class_weights)
        >>>
        >>> grads = jax.grad(loss_fn)(classifier)  # Gradients for class_weights
    """

    # Optional specifications only
    input_spec: Optional[Type[Any]] = None
    output_spec: Optional[Type[Any]] = None

    def forward(self, input: Any) -> Any:
        """Process input and return output. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Process input through the operator with optional validation."""
        class_input_spec, class_output_spec = _resolve_operator_specs(type(self))  # type: ignore[arg-type]
        input_validator = _effective_spec(self, "input_spec", class_input_spec)
        output_validator = _effective_spec(self, "output_spec", class_output_spec)

        # Use instance signature if set (e.g., by @op decorator), else resolve from forward()
        signature = self.__dict__.get("_function_signature") or _resolve_forward_signature(
            type(self)  # type: ignore[arg-type]
        )
        bound = _bind_call_arguments(signature, args, kwargs, type(self).__name__)
        single_param_name = _single_parameter_name(signature)

        if input_validator is not None:
            if single_param_name is not None:
                value = bound.arguments[single_param_name]
                bound.arguments[single_param_name] = self._validate_single_input(
                    input_validator, value
                )
            else:
                validated_arguments = self._validate_multi_args(
                    signature, input_validator, bound.arguments
                )
                bound.arguments.update(validated_arguments)

        call_args, call_kwargs = _arguments_from_mapping(signature, bound.arguments)
        output = self.forward(*call_args, **call_kwargs)

        return self._validate_output(output_validator, output)

    @staticmethod
    def _validate_single_input(validator: Type[Any], value: Any) -> Any:
        """Validate value against a Pydantic model. Caller ensures validator is valid."""
        if isinstance(value, dict):
            return validator.model_validate(value)  # type: ignore[attr-defined]
        if not isinstance(value, validator):
            return validator.model_validate(value)  # type: ignore[attr-defined]
        return value

    @staticmethod
    def _validate_output(validator: Optional[Type[Any]], output: Any) -> Any:
        """Validate output if validator is provided."""
        if validator is None:
            return output
        if isinstance(output, dict):
            return validator.model_validate(output)  # type: ignore[attr-defined]
        if not isinstance(output, validator):
            return validator.model_validate(output)  # type: ignore[attr-defined]
        return output

    def _validate_multi_args(
        self,
        signature: inspect.Signature,
        validator: Type[Any],
        arguments: OrderedDict[str, Any],
    ) -> OrderedDict[str, Any]:
        """Validate multi-argument payloads against ``input_spec``.

        Args:
            signature: Sanitized signature for ``forward`` without ``self``.
            validator: Pydantic-compatible model used for validation.
            arguments: Bound arguments being prepared for ``forward``.

        Returns:
            OrderedDict[str, Any]: Validated arguments aligned with the signature.

        Raises:
            TypeError: If ``forward`` uses variadic parameters, validation provides
                unknown fields, or the schema omits required parameters.
        """
        for param in signature.parameters.values():
            if param.kind is inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(
                    "Validation for operator callables does not support *args parameters."
                )
            if param.kind is inspect.Parameter.VAR_KEYWORD:
                raise TypeError(
                    "Validation for operator callables does not support **kwargs parameters."
                )

        payload = validator.model_validate(dict(arguments))  # type: ignore[attr-defined]
        mapping = self._coerce_mapping(payload)

        combined = OrderedDict(arguments)
        for name, value in mapping.items():
            if name not in signature.parameters:
                raise TypeError(
                    "Validation schema produced value for unknown parameter '{name}'.".format(
                        name=name
                    )
                )
            combined[name] = value

        missing = [
            param.name for param in signature.parameters.values() if param.name not in combined
        ]
        if missing:
            raise TypeError(
                "Validation did not supply values for parameter(s): {missing}.".format(
                    missing=", ".join(missing)
                )
            )

        return combined

    @staticmethod
    def _coerce_mapping(value: Any) -> Dict[str, Any]:
        """Coerce a validated Pydantic model into a mapping.

        Args:
            value: Pydantic model returned from validation.

        Returns:
            Mapping representation of the model.

        Raises:
            TypeError: If value is not a dict or Pydantic model.
        """
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            result = value.model_dump()
            if not isinstance(result, dict):
                raise TypeError(
                    f"Pydantic model_dump() returned {type(result).__name__}, expected dict"
                )
            return result
        raise TypeError(f"Expected dict or Pydantic model, got {type(value).__name__}")

    def update_params(self, **params: Any) -> "Operator":
        """Create a new operator with updated parameters (immutable update).

        Args:
            **params: Parameter names mapped to new values.

        Returns:
            New Operator instance with updated parameters.

        Raises:
            AttributeError: If a parameter name doesn't exist on the operator.
        """
        if not params:
            return self

        missing = [name for name in params if not hasattr(self, name)]
        if missing:
            annotations = getattr(type(self), "__annotations__", {})
            declared = sorted(annotations.keys()) if annotations else []
            raise AttributeError(
                f"update_params: unknown field(s): {', '.join(missing)}. "
                f"Declared fields: {', '.join(declared)}"
            )

        names = tuple(params.keys())
        result = eqx.tree_at(
            lambda op: tuple(getattr(op, name) for name in names),
            self,
            tuple(params.values()),
        )
        if not isinstance(result, Operator):
            raise TypeError(
                f"eqx.tree_at returned unexpected type {type(result).__name__}, expected Operator"
            )
        return result


__all__ = ["Operator"]


@lru_cache(maxsize=1024)
def _resolve_forward_signature(cls: Type["Operator"]) -> inspect.Signature:  # type: ignore[type-arg]
    """Return the call signature for ``cls.forward`` without ``self``."""

    forward = cls.forward
    signature = inspect.signature(forward)
    parameters = list(signature.parameters.values())

    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]

    return signature.replace(parameters=parameters)


def _single_parameter_name(signature: inspect.Signature) -> Optional[str]:
    """Return the name of the sole non-variadic parameter, if any."""

    non_variadic = [
        param
        for param in signature.parameters.values()
        if param.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]

    if len(non_variadic) == 1:
        return non_variadic[0].name
    return None


def _bind_call_arguments(
    signature: inspect.Signature,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    operator_name: str,
) -> inspect.BoundArguments:
    """Bind call arguments to a signature, applying defaults.

    Args:
        signature: Canonical signature for ``forward`` without ``self``.
        args: Positional arguments provided by the caller.
        kwargs: Keyword arguments provided by the caller.
        operator_name: Name used in error messages for context.

    Returns:
        inspect.BoundArguments: Arguments bound to the signature with defaults applied.

    Raises:
        TypeError: If argument binding fails.
    """

    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(
            f"{operator_name} received arguments incompatible with its forward signature: {exc}"
        ) from exc

    bound.apply_defaults()
    return bound


def _arguments_from_mapping(
    signature: inspect.Signature, mapping: "OrderedDict[str, Any]"
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Convert a parameter mapping into positional and keyword arguments.

    Args:
        signature: Canonical signature for ``forward`` without ``self``.
        mapping: Ordered mapping of parameter names to argument values.

    Returns:
        Tuple containing positional arguments and keyword arguments.

    Raises:
        TypeError: If required parameters are missing or variadic entries have
            incompatible container types.
    """

    positional: list[Any] = []
    keyword: Dict[str, Any] = {}

    for param in signature.parameters.values():
        name = param.name
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if name not in mapping:
                raise TypeError(f"Missing value for parameter '{name}'.")
            positional.append(mapping[name])
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            values = mapping.get(name, ())
            if not isinstance(values, tuple):
                raise TypeError(
                    "Value for variadic parameter "
                    f"'{name}' must be a tuple; received {type(values).__name__}."
                )
            positional.extend(values)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            if name not in mapping:
                raise TypeError(f"Missing value for keyword-only parameter '{name}'.")
            keyword[name] = mapping[name]
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            values = mapping.get(name, {})
            if not isinstance(values, dict):
                raise TypeError(
                    "Value for keyword variadic parameter "
                    f"'{name}' must be a dict; received {type(values).__name__}."
                )
            keyword.update(values)

    return tuple(positional), keyword


@lru_cache(maxsize=1024)
def _resolve_operator_specs(
    cls: Type["Operator"],
) -> Tuple[Optional[Type[Any]], Optional[Type[Any]]]:
    """Resolve validation specs declared on ``cls`` or via resolvable type hints."""

    input_validator = getattr(cls, "input_spec", None)
    output_validator = getattr(cls, "output_spec", None)

    if input_validator is None or output_validator is None:
        forward_fn = cls.forward
        try:
            hints = get_type_hints(forward_fn, globalns=getattr(forward_fn, "__globals__", None))
        except Exception as exc:
            missing = []
            if input_validator is None:
                missing.append("input_spec")
            if output_validator is None:
                missing.append("output_spec")
            if missing:
                joined = ", ".join(missing)
                raise TypeError(
                    f"{cls.__qualname__} must declare {joined} or use resolvable type annotations"
                ) from exc
            hints = {}

        if input_validator is None:
            hint = hints.get("input")
            if hasattr(hint, "model_validate"):
                input_validator = hint
        if output_validator is None:
            hint = hints.get("return")
            if hasattr(hint, "model_validate"):
                output_validator = hint

    return input_validator, output_validator
