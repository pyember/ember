"""The oracle's model, sent through `ember.models` and the real client.

`ClientModel` offers the same interface as `fakemodel.FakeModel`, but every
call goes through `ember.models` and the client: catalog resolution, the
limiter, `runtime.dispatch`, the attempt loop and `runtime.deferred()`. Only
the provider is fake: a `FakeProvider` whose ``respond`` answers each request
from its prompt, on the thread that serves it, so answers do not depend on
arrival order. A prompt starting with ``fail:`` fails with
``ModelError(BAD_REQUEST)``, which the client does not retry, so each call
site still sends exactly one request.
"""

import collections
import threading
import time
from collections.abc import Callable, Iterator

from fakemodel import FAIL_PREFIX, echo

import ember
from ember.errors import ErrorKind
from ember.llm import Client, Completion, ModelRef, Request, TextDelta, TextPart, using
from ember.llm.fake import Answer, Fail, FakeProvider, Reply

MODEL = "openai/gpt-5.5"


class ClientModel:
    """Calls `ember.models` with a client whose every provider is a `FakeProvider`."""

    def __init__(
        self,
        latency_s: float,
        answer: Callable[[str], str] = echo,
        *,
        arrives_before_return: bool = False,
    ) -> None:
        self._latency_s = latency_s
        self._answer = answer
        self._arrives_before_return = arrives_before_return
        self._lock = threading.Lock()
        self._reached = 0
        self._served = 0
        self._early: list[Request] = []
        self._provider = FakeProvider(respond=self._respond)
        self._client = Client(provider_factory=self._provider.factory)

    def __call__(self, prompt: str, *, system: str | None = None) -> Completion:
        self._note_reached()
        with using(self._client):
            completion = ember.models(MODEL, prompt, system=system)
        while self._arrives_before_return and not completion.done():
            time.sleep(0.001)
        return completion

    def stream(self, prompt: str) -> Iterator[str]:
        """Streams the answer to ``prompt`` as text pieces."""
        self._note_reached()
        with using(self._client):
            events = ember.models.stream(MODEL, prompt)
        return (event.text for event in events if isinstance(event, TextDelta))

    def requests(self) -> collections.Counter[str]:
        """The multiset of requests the provider received."""
        return collections.Counter(repr(seen.request) for seen in self._provider.requests)

    def reached(self) -> int:
        """How many call sites the program reached."""
        with self._lock:
            return self._reached

    def early_requests(self) -> list[Request]:
        """Requests received when no call site was waiting for one."""
        with self._lock:
            return list(self._early)

    def close(self) -> None:
        """Closes the client and its providers."""
        self._client.close()

    def _note_reached(self) -> None:
        with self._lock:
            self._reached += 1

    def _respond(self, model: ModelRef, request: Request) -> Answer:
        del model
        with self._lock:
            self._served += 1
            if self._served > self._reached:
                self._early.append(request)
        part = request.messages[-1].parts[0]
        assert isinstance(part, TextPart)
        if part.text.startswith(FAIL_PREFIX):
            return Fail(ErrorKind.BAD_REQUEST, latency_s=self._latency_s)
        return Reply(self._answer(part.text), latency_s=self._latency_s)
