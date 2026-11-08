"""The client pipeline: checks before sending, retries, pricing, reports and placement."""

import gc
import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future

import pytest

import ember
from ember import context
from ember.context import RetryPolicy
from ember.errors import ConfigError, ErrorKind, ModelError
from ember.llm import Client, Completion, Message, using
from ember.llm.attempt import retry_delay
from ember.llm.fake import Fail, FakeProvider, Reply
from ember.llm.limiter import LimiterState
from ember.llm.observe import Attempt, observing
from ember.llm.provider import StreamItem
from ember.llm.ref import ModelRef
from ember.llm.request import Request
from ember.llm.result import Done, Event, Money, Result, TextDelta, Usage, UsageEvent
from ember.llm.spec import ModelSpec
from ember.xcs import runtime


def _wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.001)


class Recorder:
    def __init__(self) -> None:
        self.attempts: list[Attempt] = []

    def on_attempt(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)


def _rate_limited(retry_after_s: float) -> Fail:
    return Fail(ErrorKind.RATE_LIMITED, status=429, retry_after_s=retry_after_s)


def _complete(client: Client, model: str, **options: object) -> Iterator[Event]:
    client.complete(model, "hi", **options)  # type: ignore[arg-type]
    return iter(())


def _stream(client: Client, model: str, **options: object) -> Iterator[Event]:
    return client.stream(model, "hi", **options)  # type: ignore[arg-type]


SURFACES = [pytest.param(_complete, id="complete"), pytest.param(_stream, id="stream")]
Surface = Callable[..., Iterator[Event]]


# Results and pricing.


def test_result_reports_the_canonical_model_and_sends_the_pinned_snapshot() -> None:
    fake = FakeProvider(default=Reply("Paris", request_id="req_1"))
    with Client(provider_factory=fake.factory) as client:
        result = client.complete("gpt-5.5", "Capital of France?").result()
    assert (result.text, result.model, result.request_id) == (
        "Paris",
        ModelRef("openai", "gpt-5.5"),
        "req_1",
    )
    assert [seen.wire_id for seen in fake.requests] == ["gpt-5.5-2026-04-23"]


def test_cost_prices_every_token_kind_exactly() -> None:
    # OpenAI's pricing page, GPT-5.5: $5.00 input, $0.50 cached input, $30.00
    # output per million tokens. 3*5 + 8*0.50 + 2*30 = 79 dollars per million.
    fake = FakeProvider(default=Reply(usage=Usage(input=3, cache_read=8, output=2, reasoning=1)))
    with Client(provider_factory=fake.factory) as client:
        assert client.complete("gpt-5.5", "hi").cost == Money(79_000)


def test_a_few_tokens_cost_nano_dollars_not_zero() -> None:
    # GPT-6 Luna input is $0.10 per million tokens: 3 tokens cost 300 nano-dollars.
    fake = FakeProvider(default=Reply(usage=Usage(input=3)))
    with Client(provider_factory=fake.factory) as client:
        assert client.complete("gpt-6-luna", "hi").cost == Money(300)


def test_provider_reported_cost_wins_over_catalog_rates() -> None:
    fake = FakeProvider(default=Reply(usage=Usage(input=1_000_000), cost=Money(123)))
    with Client(provider_factory=fake.factory) as client:
        assert client.complete("gpt-5.5", "hi").cost == Money(123)


def test_a_paid_call_to_an_unpriced_model_keeps_its_answer() -> None:
    fake = FakeProvider(default=Reply("answer", usage=Usage(input=10, output=5)))
    with Client(provider_factory=fake.factory) as client:
        result = client.complete("openai/gpt-5", "hi").result()
    assert (result.text, result.usage, result.cost) == ("answer", Usage(input=10, output=5), None)
    assert len(fake.requests) == 1


def test_unmeasured_usage_has_no_cost(client: Client) -> None:
    assert client.complete("gpt-5.5", "hi").cost is None


# Checks that happen before anything is sent, on both surfaces.


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(
    ("model", "options", "error", "match"),
    [
        ("gpt5.5", {}, ModelError, "not_found"),
        ("gpt-5.5", {"temperature": 0.2}, ValueError, "does not accept temperature"),
        ("gpt-5.5", {"reasoning_effort": "minimal"}, ValueError, "no reasoning_effort 'minimal'"),
        ("gpt-5.5", {"max_output_tokens": 128_001}, ValueError, "at most 128000 output tokens"),
        ("claude-opus-5-5", {"top_p": 0.9}, ValueError, "does not accept top_p"),
    ],
)
def test_invalid_calls_raise_before_anything_is_built_or_sent(
    surface: Surface,
    model: str,
    options: dict[str, object],
    error: type[Exception],
    match: str,
    client: Client,
    fake: FakeProvider,
) -> None:
    with pytest.raises(error, match=match):
        surface(client, model, **options)
    assert (fake.requests, fake.built) == ((), ())


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(
    ("model", "options"),
    [
        ("openai/gpt-5", {}),
        ("gemini-4-argon", {}),
        ("gpt-5.5", {"extensions": {"service_tier": "priority"}}),
    ],
    ids=["uncatalogued", "no-rates", "price-changing-extension"],
)
def test_require_price_refuses_calls_whose_price_is_unknown_before_sending(
    surface: Surface, model: str, options: dict[str, object], client: Client, fake: FakeProvider
) -> None:
    with context.override(require_price=True), pytest.raises(ModelError) as caught:
        surface(client, model, **options)
    assert caught.value.kind is ErrorKind.UNPRICED
    assert (fake.requests, fake.built) == ((), ())


@pytest.mark.parametrize(
    "model", ["claude-opus-5-5", "gemini-3.1-pro-preview", "gpt-5.4-pro", "gpt-5.5-pro"]
)
def test_require_price_sends_models_priced_for_every_token_they_report(
    model: str, client: Client, fake: FakeProvider
) -> None:
    with context.override(require_price=True):
        client.complete(model, "hi", extensions={"metadata": {"run": "r1"}})
    assert len(fake.requests) == 1


@pytest.mark.parametrize(
    ("extensions", "cost"),
    [
        ({}, Money(79_000)),
        ({"metadata": {"run": "r1"}}, Money(79_000)),
        ({"service_tier": "flex"}, None),
    ],
    ids=["none", "price-neutral", "price-changing"],
)
def test_only_price_neutral_extensions_keep_a_computed_cost(
    extensions: dict[str, object], cost: Money | None
) -> None:
    fake = FakeProvider(default=Reply(usage=Usage(input=3, cache_read=8, output=2)))
    with Client(provider_factory=fake.factory) as client:
        result = client.complete("gpt-5.5", "hi", extensions=extensions)  # type: ignore[arg-type]
        assert result.cost == cost


def test_a_custom_base_url_is_unpriced_unless_the_provider_reports_a_cost() -> None:
    fake = FakeProvider([Reply(usage=Usage(input=3)), Reply(usage=Usage(input=3), cost=Money(5))])
    gateway = {"openai": {"base_url": "https://gateway.example/v1"}}
    with Client(provider_factory=fake.factory) as client, context.override(providers=gateway):
        assert client.complete("gpt-6-luna", "hi").cost is None
        assert client.complete("gpt-6-luna", "hi").cost == Money(5)


def test_require_price_refuses_a_custom_base_url_before_sending(
    client: Client, fake: FakeProvider
) -> None:
    gateway = {"openai": {"base_url": "https://gateway.example/v1"}}
    with context.override(require_price=True, providers=gateway):
        with pytest.raises(ModelError) as caught:
            client.complete("gpt-6-luna", "hi")
    assert caught.value.kind is ErrorKind.UNPRICED
    assert (fake.requests, fake.built) == ((), ())


def test_accepted_settings_are_sent(client: Client, fake: FakeProvider) -> None:
    client.complete("gpt-5.5", "hi", reasoning_effort="xhigh", max_output_tokens=128_000)
    sampling = fake.requests[0].request.sampling
    assert (sampling.reasoning_effort, sampling.max_output_tokens) == ("xhigh", 128_000)


def test_an_uncatalogued_model_passes_every_setting_to_the_provider(
    client: Client, fake: FakeProvider
) -> None:
    client.complete("openai/gpt-5", "hi", temperature=0.2, reasoning_effort="minimal")
    sampling = fake.requests[0].request.sampling
    assert (sampling.temperature, sampling.reasoning_effort) == (0.2, "minimal")


def test_default_model_is_used_when_none_is_named(client: Client, fake: FakeProvider) -> None:
    with pytest.raises(ConfigError, match="no default_model"):
        client.complete(None, "hi")
    with context.override(default_model="claude-haiku-5-5"):
        client.complete(None, "hi")
    assert [seen.model for seen in fake.requests] == [ModelRef("anthropic", "claude-haiku-5-5")]


def test_a_conversation_prompt_is_sent_as_given(client: Client, fake: FakeProvider) -> None:
    conversation = [Message.user("hi"), Message.assistant("hello"), Message.user("bye")]
    client.complete("gpt-5.5", conversation)
    assert fake.requests[0].request.messages == tuple(conversation)


@pytest.mark.parametrize(
    ("prompt", "options", "match"),
    [
        ("hi", {"stop": "END"}, "stop must be a sequence"),
        (["hi"], {}, "prompt must be text or Messages"),
    ],
)
def test_malformed_arguments_raise_type_errors(
    prompt: object, options: dict[str, object], match: str, client: Client
) -> None:
    with pytest.raises(TypeError, match=match):
        client.complete("gpt-5.5", prompt, **options)  # type: ignore[arg-type]


# Credentials come from the context at each call.


def test_provider_overrides_take_effect_for_ember_models(
    client: Client, fake: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-0001")
    with using(client):
        ember.models("gpt-5.5", "hi")
        with context.override(providers={"openai": {"api_key": "sk-override-0002"}}):
            ember.models("gpt-5.5", "hi")
            ember.models("openai/gpt-5", "hi")
        ember.models("gpt-5.5", "hi")
    keys = [settings.credential().reveal() for settings in fake.built]  # type: ignore[union-attr]
    assert keys == ["sk-from-env-0001", "sk-override-0002"]
    assert len(fake.requests) == 4


def test_overridden_base_url_builds_its_own_provider(client: Client, fake: FakeProvider) -> None:
    client.complete("gpt-5.5", "hi")
    with context.override(providers={"openai": {"base_url": "https://gateway.example/v1"}}):
        client.complete("gpt-5.5", "hi")
    assert [settings.base_url for settings in fake.built] == [None, "https://gateway.example/v1"]


# Retries.


def test_retryable_failures_are_retried_with_backoff(sleeps: list[float]) -> None:
    fake = FakeProvider([Fail(ErrorKind.RATE_LIMITED), Fail(ErrorKind.OVERLOADED), Reply("done")])
    recorder = Recorder()
    with Client(provider_factory=fake.factory, sleep=sleeps.append, jitter=lambda: 0.5) as client:
        with observing(recorder):
            assert client.complete("gpt-5.5", "hi").text == "done"
    assert sleeps == [0.5, 1.0]
    outcomes = [
        (a.number, a.will_retry, a.outcome.kind if isinstance(a.outcome, ModelError) else "ok")
        for a in recorder.attempts
    ]
    assert outcomes == [
        (1, True, ErrorKind.RATE_LIMITED),
        (2, True, ErrorKind.OVERLOADED),
        (3, False, "ok"),
    ]
    assert len({a.call_id for a in recorder.attempts}) == 1


@pytest.mark.parametrize(
    "kind",
    [
        ErrorKind.AUTH,
        ErrorKind.PERMISSION,
        ErrorKind.QUOTA,
        ErrorKind.BAD_REQUEST,
        ErrorKind.NOT_FOUND,
        ErrorKind.MALFORMED,
    ],
)
def test_failures_that_cannot_succeed_on_retry_are_raised_at_once(
    kind: ErrorKind, sleeps: list[float]
) -> None:
    fake = FakeProvider([Fail(kind, status=400)])
    with Client(provider_factory=fake.factory, sleep=sleeps.append) as client:
        with pytest.raises(ModelError) as caught:
            client.complete("gpt-5.5", "hi")
    assert (caught.value.kind, caught.value.status) == (kind, 400)
    assert (len(fake.requests), sleeps) == (1, [])


def test_retries_stop_after_max_attempts(sleeps: list[float]) -> None:
    fake = FakeProvider([Fail(ErrorKind.SERVER)] * 5)
    with Client(provider_factory=fake.factory, sleep=sleeps.append, jitter=lambda: 0.5) as client:
        with pytest.raises(ModelError) as caught:
            client.complete("gpt-5.5", "hi")
    assert caught.value.kind is ErrorKind.SERVER
    assert (len(fake.requests), sleeps) == (3, [0.5, 1.0])


def test_retry_after_is_honoured(sleeps: list[float]) -> None:
    fake = FakeProvider([_rate_limited(2.0), Reply()])
    with Client(provider_factory=fake.factory, sleep=sleeps.append, jitter=lambda: 0.5) as client:
        client.complete("gpt-5.5", "hi")
    assert sleeps == [2.0]


@pytest.mark.parametrize("retry_after_s", [20.001, 2.3e9])  # 2.3e9: "Retry-After" in 2099
def test_a_retry_after_beyond_max_delay_is_raised_not_slept(
    retry_after_s: float, sleeps: list[float]
) -> None:
    fake = FakeProvider([_rate_limited(retry_after_s), Reply()])
    with Client(provider_factory=fake.factory, sleep=sleeps.append) as client:
        with pytest.raises(ModelError) as caught:
            client.complete("gpt-5.5", "hi")
    assert caught.value.retry_after_s == retry_after_s
    assert (len(fake.requests), sleeps) == (1, [])


def test_a_retry_after_at_max_delay_is_honoured(sleeps: list[float]) -> None:
    fake = FakeProvider([_rate_limited(20.0), Reply()])
    with Client(provider_factory=fake.factory, sleep=sleeps.append, jitter=lambda: 0.5) as client:
        client.complete("gpt-5.5", "hi")
    assert sleeps == [20.0]


@pytest.mark.parametrize(
    ("attempt", "jitter", "expected"),
    [(1, 0.0, 5.0), (1, 0.5, 10.0), (1, 0.99, 12.0), (2, 0.0, 10.0), (10_000, 0.0, 12.0)],
)
def test_backoff_doubles_with_jitter_then_clamps(
    attempt: int, jitter: float, expected: float
) -> None:
    policy = RetryPolicy(max_attempts=20_000, base_delay_s=10.0, max_delay_s=12.0)
    error = ModelError(ErrorKind.OVERLOADED, provider="openai", model="m", reason="busy")
    assert retry_delay(policy, attempt, error, jitter) == expected


# The limiter is wired into the call path.


def test_rate_limiting_lowers_the_providers_limit_and_permits_return() -> None:
    fake = FakeProvider([Fail(ErrorKind.RATE_LIMITED), Reply(), Fail(ErrorKind.AUTH)])
    with Client(provider_factory=fake.factory, sleep=lambda _: None) as client:
        with context.override(max_in_flight_per_provider=8):
            client.complete("gpt-5.5", "hi")
            with pytest.raises(ModelError):
                client.complete("gpt-5.5", "hi")
        assert client.limiter_state("openai") == LimiterState(limit=4, in_flight=0, waiting=0)
        assert client.limiter_state("anthropic") is None


# Where failures raise.


def test_outside_jit_a_failed_call_raises_at_the_call_site() -> None:
    fake = FakeProvider([Fail(ErrorKind.AUTH)])
    with Client(provider_factory=fake.factory) as client:
        with pytest.raises(ModelError):
            client.complete("gpt-5.5", "hi")


def test_when_deferred_a_failed_call_raises_on_first_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "deferred", lambda: True)
    fake = FakeProvider([Fail(ErrorKind.AUTH)])
    with Client(provider_factory=fake.factory) as client:
        completion = client.complete("gpt-5.5", "hi")
        with pytest.raises(ModelError) as caught:
            completion.text  # noqa: B018
        assert caught.value.kind is ErrorKind.AUTH
        with pytest.raises(ModelError, match="not_found"):
            client.complete("gpt5.5", "hi")


# Observers.


def test_observers_active_at_the_call_site_see_each_call_once(client: Client) -> None:
    outer, inner = Recorder(), Recorder()
    with observing(outer):
        client.complete("gpt-5.5", "one")
        with observing(inner):
            client.complete("gpt-5.5", "two")
    client.complete("gpt-5.5", "three")
    assert [a.number for a in outer.attempts] == [1, 1]
    assert [a.number for a in inner.attempts] == [1]
    assert inner.attempts[0].call_id == outer.attempts[1].call_id != outer.attempts[0].call_id
    assert all(isinstance(a.outcome, Result) for a in outer.attempts)


class GatedRuntime:
    """Runs dispatched tasks on a thread once released, with deferred results."""

    def __init__(self) -> None:
        self.gate = threading.Event()

    def dispatch(self, task: Callable[[], Result], *, label: str) -> "Future[Result]":
        future: Future[Result] = Future()

        def run() -> None:
            assert self.gate.wait(timeout=10)
            try:
                future.set_result(task())
            except ModelError as error:
                future.set_exception(error)

        threading.Thread(target=run).start()
        return future


@pytest.mark.parametrize(
    ("script", "fails"),
    [([Fail(ErrorKind.OVERLOADED), Reply("ok")], False), ([Fail(ErrorKind.AUTH)], True)],
    ids=["succeeds", "fails"],
)
def test_every_attempt_is_reported_before_the_outcome_is_readable(
    script: list[Fail | Reply], fails: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    gated = GatedRuntime()
    monkeypatch.setattr(runtime, "dispatch", gated.dispatch)
    monkeypatch.setattr(runtime, "deferred", lambda: True)
    pending: list[Completion] = []
    done_when_reported: list[bool] = []

    class Watch(Recorder):
        def on_attempt(self, attempt: Attempt) -> None:
            done_when_reported.append(pending[0].done())
            super().on_attempt(attempt)

    watch = Watch()
    fake = FakeProvider(script)
    with Client(provider_factory=fake.factory, sleep=lambda _: None) as client:
        with observing(watch):
            pending.append(client.complete("gpt-5.5", "hi"))
        gated.gate.set()
        if fails:
            with pytest.raises(ModelError):
                pending[0].result()
        else:
            assert pending[0].text == "ok"
        assert len(watch.attempts) == len(script)
    assert done_when_reported == [False] * len(script)


def test_a_stream_reports_its_attempts_before_done() -> None:
    recorder = Recorder()
    fake = FakeProvider([Fail(ErrorKind.OVERLOADED), Reply("ok")])
    with Client(provider_factory=fake.factory, sleep=lambda _: None) as client:
        with observing(recorder):
            for event in client.stream("gpt-5.5", "hi"):
                if isinstance(event, Done):
                    assert len(recorder.attempts) == 2


class Broken:
    def __init__(self) -> None:
        self.error = ValueError("observer bug")

    def on_attempt(self, attempt: Attempt) -> None:
        raise self.error


def test_an_observer_that_raises_keeps_the_result_and_the_other_observers(
    client: Client,
) -> None:
    broken, recorder = Broken(), Recorder()
    with pytest.raises(ExceptionGroup) as caught:
        with observing(broken), observing(recorder):
            completion = client.complete("gpt-5.5", "hi")
            assert completion.text == "ok"
            assert len(recorder.attempts) == 1
    assert caught.value.exceptions == (broken.error,)


# Streams.


def test_a_stream_yields_deltas_then_done() -> None:
    fake = FakeProvider(default=Reply("one two", usage=Usage(input=3)))
    with Client(provider_factory=fake.factory) as client:
        events = list(client.stream("gpt-6-luna", "hi"))
    assert events[:-1] == [TextDelta("one "), TextDelta("two"), UsageEvent(Usage(input=3))]
    done = events[-1]
    assert isinstance(done, Done)
    assert (done.result.text, done.result.cost) == ("one two", Money(300))


def test_a_stream_retries_before_its_first_item_but_not_after(sleeps: list[float]) -> None:
    fake = FakeProvider(
        [Fail(ErrorKind.OVERLOADED), Reply("a b", stream_error=ErrorKind.OVERLOADED)]
    )
    with Client(provider_factory=fake.factory, sleep=sleeps.append, jitter=lambda: 0.5) as client:
        received: list[Event] = []
        with pytest.raises(ModelError) as caught:
            for event in client.stream("gpt-5.5", "hi"):
                received.append(event)
        assert caught.value.kind is ErrorKind.OVERLOADED
        assert received == [TextDelta("a "), TextDelta("b")]
        assert (len(fake.requests), sleeps) == (2, [0.5])
        # Both attempts were overloaded, the second sent after the first halving.
        assert client.limiter_state("openai") == LimiterState(limit=16, in_flight=0, waiting=0)


class MislabelledStream(FakeProvider):
    """Fails mid-stream with an error that wrongly claims nothing was received."""

    streams = 0

    def stream(self, spec: ModelSpec, request: Request) -> Iterator[StreamItem]:
        self.streams += 1
        yield TextDelta("partial ")
        raise ModelError(ErrorKind.SERVER, provider="openai", model="m", reason="cut")


def test_the_client_never_replays_a_stream_that_delivered_output(sleeps: list[float]) -> None:
    fake = MislabelledStream()
    with Client(provider_factory=fake.factory, sleep=sleeps.append) as client:
        received: list[Event] = []
        with pytest.raises(ModelError):
            for event in client.stream("gpt-5.5", "hi"):
                received.append(event)
    assert received == [TextDelta("partial ")]
    assert (fake.streams, sleeps) == (1, [])


def test_a_stream_dropped_mid_loop_is_reported_cancelled_and_frees_its_permit() -> None:
    recorder = Recorder()
    fake = FakeProvider(default=Reply("one two three"))
    with Client(provider_factory=fake.factory) as client, observing(recorder):
        for _ in client.stream("gpt-5.5", "hi"):
            break
        gc.collect()
        assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=0, waiting=0)
    assert len(fake.requests) == 1
    [attempt] = recorder.attempts
    assert isinstance(attempt.outcome, ModelError)
    assert attempt.outcome.kind is ErrorKind.CANCELLED


def test_an_abandoned_stream_is_reported_cancelled_and_frees_its_permit() -> None:
    recorder = Recorder()
    fake = FakeProvider(default=Reply("one two three"))
    with Client(provider_factory=fake.factory) as client, observing(recorder):
        events = client.stream("gpt-5.5", "hi")
        assert next(events) == TextDelta("one ")
        events.close()  # type: ignore[attr-defined]
        assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=0, waiting=0)
    [attempt] = recorder.attempts
    assert isinstance(attempt.outcome, ModelError)
    assert attempt.outcome.kind is ErrorKind.CANCELLED


def test_reading_to_done_and_stopping_is_not_a_cancellation(client: Client) -> None:
    recorder = Recorder()
    with observing(recorder):
        events = client.stream("gpt-5.5", "hi")
        for event in events:
            if isinstance(event, Done):
                break
        events.close()  # type: ignore[attr-defined]
    assert [type(a.outcome) for a in recorder.attempts] == [Result]


def test_a_stream_is_sent_at_its_call_site_even_if_never_read(
    client: Client, fake: FakeProvider
) -> None:
    events = client.stream("gpt-5.5", "hi")
    assert len(fake.requests) == 1
    assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=1, waiting=0)
    del events
    gc.collect()
    assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=0, waiting=0)
    assert len(fake.requests) == 1


def test_outside_jit_a_failure_before_the_first_event_raises_at_the_call_site() -> None:
    fake = FakeProvider([Fail(ErrorKind.AUTH)])
    with Client(provider_factory=fake.factory) as client:
        with pytest.raises(ModelError) as caught:
            client.stream("gpt-5.5", "hi")
        assert caught.value.kind is ErrorKind.AUTH
        assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=0, waiting=0)


def test_when_deferred_a_failure_before_the_first_event_raises_on_first_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "deferred", lambda: True)
    fake = FakeProvider([Fail(ErrorKind.AUTH)])
    with Client(provider_factory=fake.factory) as client:
        events = client.stream("gpt-5.5", "hi")
        with pytest.raises(ModelError) as caught:
            next(events)
        assert caught.value.kind is ErrorKind.AUTH


@pytest.mark.parametrize(
    ("surface", "seam"), [(_complete, "dispatch"), (_stream, "dispatch_stream")]
)
def test_a_dispatch_that_cannot_start_returns_the_permit(
    surface: Surface, seam: str, client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = getattr(runtime, seam)
    failures = iter([RuntimeError("can't start new thread")])

    def start_or_fail(task: Callable[[], object], *, label: str) -> object:
        failure = next(failures, None)
        if failure is not None:
            raise failure
        return real(task, label=label)

    monkeypatch.setattr(runtime, seam, start_or_fail)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        surface(client, "gpt-5.5")
    assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=0, waiting=0)
    list(surface(client, "gpt-5.5"))
    assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=0, waiting=0)


# A call that only its own caller could unblock.


@pytest.mark.parametrize("surface", SURFACES)
def test_a_call_that_would_wait_on_its_own_open_stream_raises_before_sending(
    surface: Surface, client: Client, fake: FakeProvider
) -> None:
    with context.override(max_in_flight_per_provider=1):
        events = client.stream("gpt-5.5", "first")
        assert next(events) == TextDelta("ok")
        with pytest.raises(ModelError) as caught:
            surface(client, "gpt-5.5")
        assert caught.value.kind is ErrorKind.BAD_REQUEST
        assert "would wait forever" in str(caught.value)
        assert len(fake.requests) == 1
        events.close()  # type: ignore[attr-defined]
        assert client.limiter_state("openai") == LimiterState(limit=1, in_flight=0, waiting=0)


def test_another_threads_open_stream_makes_a_call_wait_not_fail(
    client: Client, fake: FakeProvider
) -> None:
    opened, finish = threading.Event(), threading.Event()
    texts: list[str] = []

    def hold_a_stream() -> None:
        with context.override(max_in_flight_per_provider=1):
            events = client.stream("gpt-5.5", "first")
            next(events)
            opened.set()
            assert finish.wait(timeout=10)
            list(events)

    def call() -> None:
        with context.override(max_in_flight_per_provider=1):
            texts.append(client.complete("gpt-5.5", "second").text)

    holder = threading.Thread(target=hold_a_stream)
    holder.start()
    assert opened.wait(timeout=10)
    caller = threading.Thread(target=call)
    caller.start()
    _wait_for(lambda: client.limiter_state("openai") == LimiterState(1, 1, 1))
    finish.set()
    holder.join(timeout=10)
    caller.join(timeout=10)
    assert texts == ["ok"]


# Lifetime.


def test_close_closes_providers_and_refuses_new_calls(fake: FakeProvider) -> None:
    client = Client(provider_factory=fake.factory)
    client.complete("gpt-5.5", "hi")
    client.complete("claude-opus-5-5", "hi")
    client.close()
    assert fake.closes == 2
    with pytest.raises(RuntimeError, match="closed"):
        client.complete("gpt-5.5", "hi")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="the platform has no fork")
def test_a_forked_child_builds_its_own_providers(client: Client, fake: FakeProvider) -> None:
    client.complete("gpt-5.5", "hi")
    pid = os.fork()
    if pid == 0:
        try:
            client.complete("gpt-5.5", "hi")
            healthy = len(fake.built) == 2 and fake.closes == 0
        finally:
            os._exit(0 if healthy else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert len(fake.built) == 1


def test_hot_path_overhead_with_a_zero_latency_provider(
    client: Client, capsys: pytest.CaptureFixture[str]
) -> None:
    for _ in range(200):
        client.complete("gpt-5.5", "hi")
    calls = 2_000
    started = time.perf_counter()
    for _ in range(calls):
        client.complete("gpt-5.5", "hi")
    per_call_us = (time.perf_counter() - started) / calls * 1e6
    with capsys.disabled():
        print(f"\nclient overhead, zero-latency fake provider: {per_call_us:.1f} us per call")
    assert per_call_us < 1_000
