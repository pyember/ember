"""The client pipeline: checks before sending, retries, pricing, reports and placement."""

import gc
import os
import time
from collections.abc import Callable, Iterator

import pytest

import ember
from ember import context
from ember.context import RetryPolicy
from ember.errors import ConfigError, ErrorKind, ModelError
from ember.llm import Client, Message, using
from ember.llm.attempt import retry_delay
from ember.llm.fake import FakeProvider, Reply
from ember.llm.limiter import LimiterState
from ember.llm.observe import Attempt, observing
from ember.llm.ref import ModelRef
from ember.llm.result import Done, Event, Money, Result, TextDelta, Usage, UsageEvent
from ember.xcs import runtime


class Recorder:
    def __init__(self) -> None:
        self.attempts: list[Attempt] = []

    def on_attempt(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)


def _rate_limited(retry_after_s: float) -> ModelError:
    return ModelError(
        ErrorKind.RATE_LIMITED,
        provider="openai",
        model="gpt-5.5",
        reason="slow down",
        status=429,
        retry_after_s=retry_after_s,
    )


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
@pytest.mark.parametrize("model", ["openai/gpt-5", "gemini-4-argon", "gemini-3.1-pro-preview"])
def test_require_price_refuses_models_without_a_full_price_before_sending(
    surface: Surface, model: str, client: Client, fake: FakeProvider
) -> None:
    with context.override(require_price=True), pytest.raises(ModelError) as caught:
        surface(client, model)
    assert caught.value.kind is ErrorKind.UNPRICED
    assert (fake.requests, fake.built) == ((), ())


def test_require_price_sends_fully_priced_models(client: Client, fake: FakeProvider) -> None:
    with context.override(require_price=True):
        client.complete("claude-opus-5-5", "hi")
    assert len(fake.requests) == 1


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
    fake = FakeProvider([ErrorKind.RATE_LIMITED, ErrorKind.OVERLOADED, Reply("done")])
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
    "failure",
    [
        ErrorKind.AUTH,
        ErrorKind.BAD_REQUEST,
        ModelError(
            ErrorKind.TRANSPORT, provider="openai", model="m", reason="cut", before_first_byte=False
        ),
    ],
)
def test_failures_that_cannot_succeed_on_retry_are_raised_at_once(
    failure: ErrorKind | ModelError, sleeps: list[float]
) -> None:
    fake = FakeProvider([failure])
    with Client(provider_factory=fake.factory, sleep=sleeps.append) as client:
        with pytest.raises(ModelError) as caught:
            client.complete("gpt-5.5", "hi")
    expected = failure if isinstance(failure, ErrorKind) else failure.kind
    assert caught.value.kind is expected
    assert (len(fake.requests), sleeps) == (1, [])


def test_retries_stop_after_max_attempts(sleeps: list[float]) -> None:
    fake = FakeProvider([ErrorKind.SERVER] * 5)
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


def test_a_retry_after_beyond_max_delay_is_not_retried(sleeps: list[float]) -> None:
    fake = FakeProvider([_rate_limited(21.0), Reply()])
    with Client(provider_factory=fake.factory, sleep=sleeps.append) as client:
        with pytest.raises(ModelError) as caught:
            client.complete("gpt-5.5", "hi")
    assert caught.value.retry_after_s == 21.0
    assert (len(fake.requests), sleeps) == (1, [])


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
    fake = FakeProvider([ErrorKind.RATE_LIMITED, Reply(), ErrorKind.AUTH])
    with Client(provider_factory=fake.factory, sleep=lambda _: None) as client:
        with context.override(max_in_flight_per_provider=8):
            client.complete("gpt-5.5", "hi")
            with pytest.raises(ModelError):
                client.complete("gpt-5.5", "hi")
        assert client.limiter_state("openai") == LimiterState(limit=4, in_flight=0, waiting=0)
        assert client.limiter_state("anthropic") is None


# Where failures raise.


def test_outside_jit_a_failed_call_raises_at_the_call_site() -> None:
    fake = FakeProvider([ErrorKind.AUTH])
    with Client(provider_factory=fake.factory) as client:
        with pytest.raises(ModelError):
            client.complete("gpt-5.5", "hi")


def test_when_deferred_a_failed_call_raises_on_first_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "deferred", lambda: True)
    fake = FakeProvider([ErrorKind.AUTH])
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


# Streams.


def test_a_stream_yields_deltas_then_done() -> None:
    fake = FakeProvider(default=Reply("one two", usage=Usage(input=3)))
    with Client(provider_factory=fake.factory) as client:
        events = list(client.stream("gpt-6-luna", "hi"))
    assert events[:-1] == [TextDelta("one "), TextDelta("two"), UsageEvent(Usage(input=3))]
    done = events[-1]
    assert isinstance(done, Done)
    assert (done.result.text, done.result.cost) == ("one two", Money(300))


def test_a_stream_retries_only_before_its_first_item(sleeps: list[float]) -> None:
    fake = FakeProvider([ErrorKind.OVERLOADED, Reply("a b", stream_error=ErrorKind.OVERLOADED)])
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


def test_a_stream_never_read_still_frees_its_permit(client: Client, fake: FakeProvider) -> None:
    events = client.stream("gpt-5.5", "hi")
    assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=1, waiting=0)
    del events
    gc.collect()
    assert client.limiter_state("openai") == LimiterState(limit=64, in_flight=0, waiting=0)
    assert fake.requests == ()


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
