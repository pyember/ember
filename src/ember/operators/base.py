"""Operators, their composition with ``>>``, and `op`.

An operator is a frozen, slotted dataclass that implements
``forward(self, input: In) -> Out``. Calling an operator validates its input,
runs ``forward`` and validates its output, where the class declares
validation; ``dataclasses.replace`` is the only way to change one.
``first >> second`` builds a `Chain` and type-checks that ``second`` takes
what ``first`` returns.

Validation is declared in one place: the ``input_model`` and ``output_model``
class attributes, each a pydantic model with ``extra="forbid"``. They are
checked when the class is defined and applied in strict mode on every call.
Nothing is read from parameter names or annotations, so operators behave the
same under ``from __future__ import annotations`` and ``TYPE_CHECKING`` imports.

Example:
    >>> import dataclasses
    >>> from ember.operators import Operator
    >>> @dataclasses.dataclass(frozen=True, slots=True)
    ... class Exclaim(Operator[str, str]):
    ...     mark: str = "!"
    ...     def forward(self, input: str) -> str:
    ...         return input + self.mark
    >>> Exclaim()("hi")
    'hi!'
    >>> dataclasses.replace(Exclaim(), mark="?")("hi")
    'hi?'
"""

import dataclasses
import inspect
import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = ["Chain", "FunctionOperator", "Operator", "op"]

In = TypeVar("In", contravariant=True)
Out = TypeVar("Out", covariant=True)
T = TypeVar("T")
R = TypeVar("R")


def _require_frozen_slotted_dataclass(cls: type) -> None:
    for klass in cls.__mro__:
        if klass is Operator:
            return
        if "__dataclass_fields__" not in klass.__dict__ or "__slots__" not in klass.__dict__:
            raise TypeError(
                f"{klass.__qualname__} must be decorated with "
                "@dataclasses.dataclass(frozen=True, slots=True)"
            )


def _require_boundary_model(cls: type, name: str, model: object) -> None:
    if model is None:
        return
    # Imported here so that operators without validation never load pydantic.
    import pydantic

    if not (isinstance(model, type) and issubclass(model, pydantic.BaseModel)):
        raise TypeError(f"{cls.__qualname__}.{name} must be a pydantic model class")
    if model.model_config.get("extra") != "forbid":
        raise TypeError(
            f"{cls.__qualname__}.{name}: {model.__qualname__} must set "
            "model_config = ConfigDict(extra='forbid')"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Operator(Generic[In, Out]):
    """A typed, immutable unit of computation.

    Subclasses are decorated with ``@dataclasses.dataclass(frozen=True,
    slots=True)`` and implement `forward`. Constructing an instance of a class
    that is not decorated that way raises `TypeError`.

    Attributes:
        input_model: Optional pydantic model the input is validated against,
            strictly, before `forward` runs. A mapping of its fields is
            validated into an instance; an instance passes through.
        output_model: Optional pydantic model the output of `forward` is
            validated against, strictly, before it is returned.
    """

    input_model: ClassVar["type[BaseModel] | None"] = None
    output_model: ClassVar["type[BaseModel] | None"] = None

    # `slots=True` replaces the class, which breaks zero-argument super() before
    # Python 3.14, so these two methods name their targets explicitly.
    def __new__(cls, *args: object, **kwargs: object) -> "Operator[In, Out]":
        del args, kwargs  # Consumed by the dataclass __init__.
        _require_frozen_slotted_dataclass(cls)
        return object.__new__(cls)

    def __init_subclass__(cls) -> None:
        super(Operator, cls).__init_subclass__()
        _require_boundary_model(cls, "input_model", cls.input_model)
        _require_boundary_model(cls, "output_model", cls.output_model)

    def forward(self, input: In) -> Out:
        """Computes the output for one input. Subclasses implement this."""
        raise NotImplementedError(f"{type(self).__qualname__} does not implement forward")

    def __call__(self, input: In) -> Out:
        """Validates ``input``, runs `forward`, and validates its output.

        Raises:
            pydantic.ValidationError: The input or output does not satisfy the
                declared model.
        """
        if self.input_model is not None:
            input = cast(In, self.input_model.model_validate(input, strict=True))
        output = self.forward(input)
        if self.output_model is not None:
            output = cast(Out, self.output_model.model_validate(output, strict=True))
        return output

    def __rshift__(self, following: "Operator[Out, R]") -> "Chain[In, R]":
        """Returns a `Chain` that runs this operator, then ``following``."""
        if not isinstance(following, Operator):
            return NotImplemented
        return Chain(_stages_of(self) + _stages_of(following))


@dataclasses.dataclass(frozen=True, slots=True)
class Chain(Operator[In, Out]):
    """Feeds its input to the first stage and each output to the next stage.

    Build one with ``>>``, which checks adjacent types statically. Where
    adjacent stages both declare validation models, construction also checks
    that each stage's ``output_model`` is the next stage's ``input_model`` or a
    subclass of it.

    Attributes:
        stages: The operators, in order; at least one. Their input and output
            types differ from stage to stage, hence ``Operator[Any, Any]``.
    """

    stages: tuple[Operator[Any, Any], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stages, tuple):
            raise TypeError(f"Chain stages must be a tuple, got {type(self.stages)}")
        if not self.stages:
            raise ValueError("a Chain needs at least one stage")
        for index, stage in enumerate(self.stages):
            if not isinstance(stage, Operator):
                raise TypeError(f"Chain stage {index} is not an Operator; wrap it with @op")
        for index, (stage, following) in enumerate(itertools.pairwise(self.stages)):
            produced, expected = stage.output_model, following.input_model
            if produced is not None and expected is not None and not issubclass(produced, expected):
                raise TypeError(
                    f"Chain stage {index} produces {produced.__qualname__} but stage {index + 1} "
                    f"takes {expected.__qualname__}"
                )

    def forward(self, input: In) -> Out:
        value: Any = input  # Each stage's output is the next stage's input.
        for stage in self.stages:
            value = stage(value)
        return cast(Out, value)


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionOperator(Operator[In, Out]):
    """An operator whose `forward` is a one-argument function. Built by `op`.

    Attributes:
        fn: The function; it is called with the operator's input.
    """

    fn: Callable[[In], Out]

    def forward(self, input: In) -> Out:
        return self.fn(input)


def op(fn: Callable[[T], R], /) -> FunctionOperator[T, R]:
    """Turns a function of one argument into an operator.

    The operator is built here, once, and never changes. Declare validation by
    subclassing `Operator` instead; `op` adds none.

    Example:
        >>> from ember import op
        >>> @op
        ... def shout(text: str) -> str:
        ...     return text.upper()
        >>> shout("hi")
        'HI'

    Raises:
        TypeError: ``fn`` cannot be called with exactly one positional argument.
    """
    try:
        inspect.signature(fn).bind(None)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"op needs a function of one positional argument: {exc}") from None
    return FunctionOperator(fn)


def _stages_of(operator: Operator[Any, Any]) -> tuple[Operator[Any, Any], ...]:
    return operator.stages if isinstance(operator, Chain) else (operator,)
