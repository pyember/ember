"""`ember.models`: call any model in one line.

Example:
    >>> import ember
    >>> from ember.llm import Client, using
    >>> from ember.llm.fake import FakeProvider, Reply
    >>> fake = FakeProvider(default=Reply("Paris"))
    >>> with Client(provider_factory=fake.factory) as client, using(client):
    ...     answer = ember.models("gpt-5.5", "Capital of France?", max_output_tokens=50)
    ...     print(answer.text, answer.model)
    Paris openai/gpt-5.5
"""

from collections.abc import Iterator, Mapping, Sequence

from ember.llm.client import BoundModel, CallOptions, ModelName, Prompt, active
from ember.llm.completion import Completion
from ember.llm.request import JsonValue, ToolChoice, ToolSpec
from ember.llm.result import Event

__all__ = ["Models", "models"]


class Models:
    """The type of `ember.models`. Calls go to `ember.llm.client.active()`."""

    __slots__ = ()

    def __call__(
        self,
        model: ModelName,
        prompt: Prompt,
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: Sequence[str] = (),
        seed: int | None = None,
        reasoning_effort: str | None = None,
        output_schema: Mapping[str, JsonValue] | None = None,
        extensions: Mapping[str, JsonValue] | None = None,
    ) -> Completion:
        """Calls ``model`` with ``prompt`` and returns its `Completion`.

        Outside ``@xcs.jit`` the call has finished when this returns; inside,
        it is in flight and reading any field waits for it.

        Args:
            model: ``provider/name``, a catalog alias such as ``"gpt-5.5"``, or
                ``None`` for the context's ``default_model``.
            prompt: The user's text, or a whole conversation of `Message`.
            system: System instructions.
            tools: Tools the model may call.
            tool_choice: ``"auto"``, ``"none"``, ``"required"``, or a tool name.
            temperature: Sampling temperature.
            top_p: Nucleus sampling mass.
            max_output_tokens: Upper bound on generated tokens, reasoning included.
            stop: Sequences that end generation.
            seed: Seed for reproducible sampling.
            reasoning_effort: One of the model's reasoning efforts.
            output_schema: JSON Schema the answer must follow.
            extensions: Provider-specific body fields, named explicitly.

        Raises:
            ModelError: ``NOT_FOUND`` for an unknown model, ``UNPRICED`` under
                ``require_price``, or the call's failure.
            ValueError: A setting the model does not accept.
            ConfigError: The provider is not configured or has no credential.
        """
        return active().complete(
            model,
            prompt,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            stop=stop,
            seed=seed,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
            extensions=extensions,
        )

    def stream(
        self,
        model: ModelName,
        prompt: Prompt,
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: Sequence[str] = (),
        seed: int | None = None,
        reasoning_effort: str | None = None,
        output_schema: Mapping[str, JsonValue] | None = None,
        extensions: Mapping[str, JsonValue] | None = None,
    ) -> Iterator[Event]:
        """Streams the answer as `TextDelta`, `ReasoningDelta`, `ToolCallDelta`
        and `UsageEvent` events, ending with `Done`. Arguments as for a call.
        """
        return active().stream(
            model,
            prompt,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            stop=stop,
            seed=seed,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
            extensions=extensions,
        )

    def bind(
        self,
        model: ModelName,
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_choice: ToolChoice = "auto",
        temperature: float | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
        stop: Sequence[str] = (),
        seed: int | None = None,
        reasoning_effort: str | None = None,
        output_schema: Mapping[str, JsonValue] | None = None,
        extensions: Mapping[str, JsonValue] | None = None,
    ) -> BoundModel:
        """Fixes the model and options; the result is called with just a prompt.

        Nothing is resolved or read until the first call, so binding at import
        time is free. Each call uses the client and context active at that call.
        """
        options = CallOptions.of(
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            stop=stop,
            seed=seed,
            reasoning_effort=reasoning_effort,
            output_schema=output_schema,
            extensions=extensions,
        )
        return BoundModel(model=model, options=options)

    def __repr__(self) -> str:
        return "ember.models"


models = Models()
