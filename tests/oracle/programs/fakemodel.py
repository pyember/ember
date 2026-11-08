"""A stand-in model that reaches the xcs scheduler exactly as the client does.

`FakeModel` builds its `Request` at the call site, hands the network work to
`ember.xcs.runtime.dispatch`, and returns the resulting `Completion`. The
network work sleeps for a fixed latency and answers deterministically, so a
plain run and a jitted run of the same program can be compared exactly.
"""

import collections
import threading
import time
from collections.abc import Callable

from ember.errors import ErrorKind, ModelError
from ember.llm import Completion, FinishReason, ModelRef, Request, Result, TextPart, Usage
from ember.xcs import runtime

REF = ModelRef("fake", "echo")
FAIL_PREFIX = "fail:"


def echo(prompt: str) -> str:
    """The default answer: the prompt in angle brackets."""
    return f"<{prompt}>"


class FakeModel:
    """Answers prompts after a fixed latency, recording every request it serves.

    A prompt starting with ``fail:`` fails with ``ModelError(RATE_LIMITED)``.
    The answer function runs on whichever thread serves the request, so it
    sees that thread's context.
    """

    def __init__(self, latency_s: float, answer: Callable[[str], str] = echo) -> None:
        self.latency_s = latency_s
        self._answer = answer
        self._lock = threading.Lock()
        self._clock = 0
        self._reached = 0
        self._served: list[tuple[int, int, Request]] = []

    def __call__(self, prompt: str, *, system: str | None = None) -> Completion:
        request = Request.from_prompt(prompt, system=system)
        with self._lock:
            self._reached += 1
        reached_at = self._tick()
        future = runtime.dispatch(lambda: self._serve(request, reached_at), label=str(REF))
        return Completion(future, REF)

    def requests(self) -> collections.Counter[str]:
        """The multiset of requests served so far."""
        with self._lock:
            return collections.Counter(repr(request) for _, _, request in self._served)

    def reached(self) -> int:
        """How many call sites the program reached."""
        with self._lock:
            return self._reached

    def early_requests(self) -> list[Request]:
        """Requests served before the program reached their call site."""
        with self._lock:
            return [request for reached, served, request in self._served if served < reached]

    def _tick(self) -> int:
        with self._lock:
            self._clock += 1
            return self._clock

    def _serve(self, request: Request, reached_at: int) -> Result:
        served_at = self._tick()
        with self._lock:
            self._served.append((reached_at, served_at, request))
        time.sleep(self.latency_s)
        part = request.messages[-1].parts[0]
        assert isinstance(part, TextPart)
        if part.text.startswith(FAIL_PREFIX):
            raise ModelError(
                ErrorKind.RATE_LIMITED, provider="fake", model="echo", reason="told to fail"
            )
        text = self._answer(part.text)
        return Result(
            text=text,
            reasoning="",
            tool_calls=(),
            finish_reason=FinishReason.STOP,
            usage=Usage(input=len(part.text), output=len(text)),
            cost=None,
            model=REF,
            request_id=None,
            latency_s=self.latency_s,
        )
