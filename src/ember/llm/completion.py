"""A model call's result, possibly still in flight.

The request behind a `Completion` was built from the caller's arguments and
handed to `ember.xcs.runtime.dispatch` before the `Completion` existed. Reading
any field waits for that request only. Outside `@xcs.jit` the call has already
finished by the time the caller sees the `Completion`.
"""

from concurrent.futures import Future

from ember.llm.ref import ModelRef
from ember.llm.result import FinishReason, Money, Result, ToolCall, Usage
from ember.xcs import runtime

__all__ = ["Completion"]


class Completion:
    """The outcome of one model call.

    Every field reads through `result()`, so each one waits until the call is
    complete and raises the call's error, unchanged, if it failed.

    ``str(completion)`` is the generated text.
    """

    __slots__ = ("_future", "_ref")

    def __init__(self, future: "Future[Result]", ref: ModelRef) -> None:
        self._future = future
        self._ref = ref

    def result(self) -> Result:
        """Waits for the call and returns its `Result`."""
        return runtime.wait(self._future)

    def done(self) -> bool:
        """Whether the call has finished, successfully or not, without waiting."""
        return self._future.done()

    @property
    def model(self) -> ModelRef:
        """The model the call was sent to; available without waiting."""
        return self._ref

    @property
    def text(self) -> str:
        """Generated text."""
        return self.result().text

    @property
    def reasoning(self) -> str:
        """Reasoning summary text, if the provider returned one."""
        return self.result().reasoning

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Tool calls the model made."""
        return self.result().tool_calls

    @property
    def finish_reason(self) -> FinishReason:
        """Why generation ended."""
        return self.result().finish_reason

    @property
    def usage(self) -> Usage:
        """Tokens consumed."""
        return self.result().usage

    @property
    def cost(self) -> Money | None:
        """Cost of the call, or ``None`` when the model's price is unknown."""
        return self.result().cost

    @property
    def request_id(self) -> str | None:
        """The provider's id for the request."""
        return self.result().request_id

    @property
    def latency_s(self) -> float:
        """Seconds from sending the request to the last byte."""
        return self.result().latency_s

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        state = "done" if self.done() else "pending"
        return f"Completion({self._ref}, {state})"
