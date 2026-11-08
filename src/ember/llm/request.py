"""The provider-neutral request.

A `Request` is validated when it is constructed. Codecs translate it to each
provider's wire format; nothing else inspects provider-specific shapes.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Literal, NewType, TypeAlias

__all__ = [
    "ImagePart",
    "JsonValue",
    "Message",
    "Part",
    "ReasoningPart",
    "Request",
    "Role",
    "Sampling",
    "TextPart",
    "ToolCallPart",
    "ToolChoice",
    "ToolName",
    "ToolResultPart",
    "ToolSpec",
    "WireApi",
]

JsonValue: TypeAlias = (
    str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
)
Role = Literal["user", "assistant", "tool"]
WireApi = Literal["openai_chat", "openai_responses", "anthropic_messages", "gemini"]
"""A provider wire protocol; each has exactly one codec."""
ToolName = NewType("ToolName", str)
"""The name of one tool the model must call."""

ToolChoice: TypeAlias = Literal["auto", "none", "required"] | ToolName


def _no_extensions() -> Mapping[str, JsonValue]:
    return MappingProxyType({})


@dataclasses.dataclass(frozen=True, slots=True)
class TextPart:
    """Plain text."""

    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class ImagePart:
    """An image given by URL or by bytes. Exactly one of the two is set."""

    media_type: str
    url: str | None = None
    data: bytes | None = None

    def __post_init__(self) -> None:
        if (self.url is None) == (self.data is None):
            raise ValueError("ImagePart needs exactly one of url or data")
        if not self.media_type.startswith("image/"):
            raise ValueError(f"ImagePart media_type must be image/*, got {self.media_type!r}")


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallPart:
    """A tool call the assistant made, replayed as conversation history."""

    id: str
    name: str
    arguments: Mapping[str, JsonValue]


@dataclasses.dataclass(frozen=True, slots=True)
class ReasoningPart:
    """Reasoning the assistant produced, kept so the conversation can continue.

    Some providers reject or degrade a tool-use turn whose earlier reasoning
    is missing from the history (Anthropic thinking blocks, Gemini thought
    signatures, OpenAI Responses reasoning items). A `Result.message` carries
    these parts in the order the provider produced them; append that message
    to the history unchanged. The codec for ``api`` sends ``state`` back
    exactly as received; codecs for other wire protocols cannot use it and
    leave the part out.

    Attributes:
        text: Readable reasoning summary; empty when the provider sent none.
        api: The wire protocol that produced the part.
        state: The provider's own data for this reasoning, opaque to callers.
    """

    text: str
    api: WireApi
    state: Mapping[str, JsonValue]


@dataclasses.dataclass(frozen=True, slots=True)
class ToolResultPart:
    """The result of executing a tool call."""

    call_id: str
    content: str
    is_error: bool = False


Part: TypeAlias = TextPart | ImagePart | ToolCallPart | ReasoningPart | ToolResultPart

_ALLOWED_PARTS: Mapping[str, tuple[type, ...]] = MappingProxyType(
    {
        "user": (TextPart, ImagePart),
        "assistant": (TextPart, ToolCallPart, ReasoningPart),
        "tool": (ToolResultPart,),
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Message:
    """One conversation turn.

    User turns hold text and images, assistant turns hold text, tool calls and
    reasoning, and tool turns hold tool results.
    """

    role: Role
    parts: tuple[Part, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError(f"a {self.role} message needs at least one part")
        allowed = _ALLOWED_PARTS[self.role]
        for part in self.parts:
            if not isinstance(part, allowed):
                raise ValueError(f"a {self.role} message cannot hold {type(part).__name__}")

    @classmethod
    def user(cls, text: str) -> "Message":
        """Returns a user message holding ``text``."""
        return cls("user", (TextPart(text),))

    @classmethod
    def assistant(cls, text: str) -> "Message":
        """Returns an assistant message holding ``text``."""
        return cls("assistant", (TextPart(text),))


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call.

    Attributes:
        name: Tool name, unique within a request.
        description: What the tool does, for the model.
        parameters: JSON Schema for the tool's arguments object.
    """

    name: str
    description: str
    parameters: Mapping[str, JsonValue]


@dataclasses.dataclass(frozen=True, slots=True)
class Sampling:
    """Generation settings. ``None`` leaves the provider's default in place.

    Attributes:
        temperature: Sampling temperature.
        top_p: Nucleus sampling mass.
        max_output_tokens: Upper bound on generated tokens, reasoning included.
        stop: Sequences that end generation.
        seed: Seed for providers that support reproducible sampling.
        reasoning_effort: Reasoning effort for models that support it; checked
            against the model's allowed efforts before sending.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")


DEFAULT_SAMPLING = Sampling()


@dataclasses.dataclass(frozen=True, slots=True)
class Request:
    """Everything sent to a model, independent of provider.

    Attributes:
        messages: The conversation, oldest first; the last turn is not an
            assistant turn.
        system: System instructions.
        tools: Tools the model may call.
        tool_choice: Whether and which tool the model must call.
        sampling: Generation settings.
        output_schema: JSON Schema the response must conform to.
        extensions: Provider-specific body fields, named explicitly by the
            caller and merged into the wire body by the codec. A key that
            collides with a field the codec sets is an error.
    """

    messages: tuple[Message, ...]
    system: str | None = None
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: ToolChoice = "auto"
    sampling: Sampling = DEFAULT_SAMPLING
    output_schema: Mapping[str, JsonValue] | None = None
    extensions: Mapping[str, JsonValue] = dataclasses.field(default_factory=_no_extensions)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a request needs at least one message")
        if self.messages[-1].role == "assistant":
            raise ValueError("the last message cannot be an assistant message")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        if self.tool_choice not in ("auto", "none", "required") and self.tool_choice not in names:
            raise ValueError(f"tool_choice names unknown tool {self.tool_choice!r}")
        if self.tool_choice == "required" and not self.tools:
            raise ValueError("tool_choice='required' needs at least one tool")

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        system: str | None = None,
        sampling: Sampling = DEFAULT_SAMPLING,
    ) -> "Request":
        """Returns a single-turn request asking ``prompt``."""
        return cls(messages=(Message.user(prompt),), system=system, sampling=sampling)
