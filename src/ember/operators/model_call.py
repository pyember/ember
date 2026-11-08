"""`ModelCall`: one model call per input."""

import dataclasses
import string
from collections.abc import Sequence
from typing import TypeAlias, TypeGuard

from ember.llm.ref import ModelRef
from ember.llm.request import Message, Sampling
from ember.llm.result import Result
from ember.operators.base import Operator
from ember.operators.upstream import models

__all__ = ["ModelCall", "Prompt"]

Prompt: TypeAlias = str | Sequence[Message]


@dataclasses.dataclass(frozen=True, slots=True)
class ModelCall(Operator[Prompt, Result]):
    """Sends its input to a model and returns the finished `Result`.

    The input is a prompt string or a sequence of `Message`; anything else
    raises `TypeError` instead of being converted to text.

    Example:
        >>> from ember.operators import ModelCall
        >>> ask = ModelCall("openai/gpt-5", template="Answer in one word: {input}")
        >>> ask.template
        'Answer in one word: {input}'

    Attributes:
        model: ``provider/name`` or a catalog alias. There is no default.
        template: A `str.format` template whose only field is ``{input}``.
            When set, the input must be a string and the prompt is the template
            with the input substituted; when ``None`` the input is sent as is.
        system: System instructions.
        sampling: Generation settings.
    """

    model: str | ModelRef
    template: str | None = None
    system: str | None = None
    sampling: Sampling = Sampling()

    def __post_init__(self) -> None:
        if isinstance(self.model, str) and not self.model.strip():
            raise ValueError("ModelCall needs a model")
        if self.template is not None:
            _require_input_template(self.template)

    def forward(self, input: Prompt) -> Result:
        sampling = self.sampling
        completion = models(
            self.model,
            self.prompt(input),
            system=self.system,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_output_tokens=sampling.max_output_tokens,
            stop=sampling.stop,
            seed=sampling.seed,
            reasoning_effort=sampling.reasoning_effort,
        )
        return completion.result()

    def prompt(self, input: Prompt) -> str | tuple[Message, ...]:
        """Returns what `forward` sends for ``input``.

        Raises:
            TypeError: ``input`` is neither a string nor a sequence of
                `Message`, or it is messages and the call has a template.
        """
        if isinstance(input, str):
            return input if self.template is None else self.template.format(input=input)
        if _is_messages(input):
            if self.template is not None:
                raise TypeError("a ModelCall with a template takes a string")
            return tuple(input)
        raise TypeError(f"ModelCall takes a string or a sequence of Message, got {type(input)}")


def _is_messages(value: object) -> TypeGuard[Sequence[Message]]:
    return isinstance(value, Sequence) and all(isinstance(item, Message) for item in value)


def _require_input_template(template: str) -> None:
    fields = [
        (name, spec, conversion)
        for _, name, spec, conversion in string.Formatter().parse(template)
        if name is not None
    ]
    if not fields:
        raise ValueError("a ModelCall template must contain {input}")
    for name, spec, conversion in fields:
        if name != "input" or spec or conversion:
            written = name + (f"!{conversion}" if conversion else "") + (f":{spec}" if spec else "")
            raise ValueError(
                f"a ModelCall template's only field is a plain {{input}}, found {{{written}}}"
            )
