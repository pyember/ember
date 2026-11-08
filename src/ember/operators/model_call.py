"""`ModelCall`: one model call per input."""

import dataclasses
import string
from collections.abc import Sequence
from typing import TypeAlias, TypeGuard

from ember.llm.api import models
from ember.llm.completion import Completion
from ember.llm.ref import ModelRef
from ember.llm.request import Message, Sampling
from ember.operators.base import Operator

__all__ = ["ModelCall", "Prompt"]

Prompt: TypeAlias = str | Sequence[Message] | Completion


@dataclasses.dataclass(frozen=True, slots=True)
class ModelCall(Operator[Prompt, Completion]):
    """Sends its input to a model and returns the call's `Completion`.

    Outside `@xcs.jit` and `xcs.map` the call has finished when `ModelCall`
    returns, and a failure raises here. Inside them the `Completion` is
    pending: the request is already sent, and reading any field waits for it
    and raises its failure. So ``a = summarize(doc); b = translate(doc)`` sends
    both requests before either is read.

    The input is a prompt string, a sequence of `Message`, or the `Completion`
    of an earlier call, whose text is the prompt (reading it waits for that
    call). Anything else raises `TypeError` instead of being converted to text.

    Example:
        >>> from ember.operators import ModelCall
        >>> ask = ModelCall("openai/gpt-5.5", template="Answer in one word: {input}")
        >>> ask.template
        'Answer in one word: {input}'

    Attributes:
        model: ``provider/name`` or a catalog alias. There is no default.
        template: A `str.format` template whose only field is ``{input}``.
            When set, the input must be text (a string or a `Completion`) and
            the prompt is the template with the text substituted; when
            ``None`` the input is sent as is.
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

    def forward(self, input: Prompt) -> Completion:
        sampling = self.sampling
        return models(
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

    def prompt(self, input: Prompt) -> str | tuple[Message, ...]:
        """Returns what `forward` sends for ``input``.

        Raises:
            TypeError: ``input`` is not a string, a sequence of `Message` or a
                `Completion`, or it is messages and the call has a template.
        """
        text = input.text if isinstance(input, Completion) else input
        if isinstance(text, str):
            return text if self.template is None else self.template.format(input=text)
        if _is_messages(text):
            if self.template is not None:
                raise TypeError("a ModelCall with a template takes text")
            return tuple(text)
        raise TypeError(
            f"ModelCall takes a string, a sequence of Message or a Completion, got {type(input)}"
        )


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
