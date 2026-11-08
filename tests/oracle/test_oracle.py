"""The xcs oracle: `jit` changes when model calls wait, never what a program computes.

Every case runs a program from `orchard` plainly and under `xcs.jit`, each
against a fresh `FakeModel`, and requires:

- the same outcome: an equal value, or an error of the same type and message;
- the same multiset of model requests, each served exactly once and never
  before the program reached its call site;
- the same side-effect log, in program order, or per item for programs whose
  items run through `xcs.map`.

Cases whose program makes N independent calls of latency L must also finish
in about L under jit, and never slower than the plain run.
"""

import collections
import dataclasses
import time
from collections.abc import Callable
from typing import Literal

import fakemodel
import orchard
import pytest

from ember import context, xcs
from ember.errors import ModelError

Program = Callable[..., object]
Transform = Callable[[Program], Program]
Drive = Callable[[Transform, fakemodel.FakeModel], object]

SEMANTIC_LATENCY_S = 0.005
OVERLAP_LATENCY_S = 0.25
# Scheduling overhead plus the jitter of a shared CI runner.
OVERHEAD_S = 0.25
# Programs that use xcs.map overlap in the plain run too; allow timer noise.
NOISE_S = 0.1


def plain(program: Program) -> Program:
    return program


@dataclasses.dataclass(frozen=True)
class Case:
    """One oracle scenario.

    Attributes:
        name: Test id.
        drive: Calls the transformed program, possibly several times.
        independent: Independent calls the program makes, for the overlap check.
        order: How side effects are compared.
        answer: How the fake answers, on the thread that serves the request.
        error: The error type both runs must raise, if any.
    """

    name: str
    drive: Drive
    independent: int = 0
    order: Literal["program", "per_item"] = "program"
    answer: Callable[[str], str] = fakemodel.echo
    error: type[Exception] | None = None


@dataclasses.dataclass(frozen=True)
class Run:
    outcome: object
    requests: collections.Counter[str]
    reached: int
    early: list[object]
    effects: object
    wall_s: float


def call(program: Program, *args: object, **kwargs: object) -> Drive:
    """Drives one call of the transformed program."""
    return lambda transform, model: transform(program)(model, *args, **kwargs)


def repeat(program: Program, times: int, *args: object) -> Drive:
    """Calls one transformed program several times; nothing may carry over."""

    def drive(transform: Transform, model: fakemodel.FakeModel) -> object:
        transformed = transform(program)
        return [transformed(model, *args) for _ in range(times)]

    return drive


def drive_persona(transform: Transform, model: fakemodel.FakeModel) -> object:
    reply = transform(orchard.persona_reply)
    answers = []
    for name in ("pirate", "lawyer", "lawyer"):
        orchard.PERSONA["name"] = name
        answers.append(reply(model, "hi"))
    return answers


def drive_aliasing(transform: Transform, model: fakemodel.FakeModel) -> object:
    aliased = transform(orchard.aliased_arguments)
    same: list[str] = []
    first = aliased(model, same, same)
    dst: list[str] = []
    src: list[str] = []
    second = aliased(model, dst, src)
    return [first, same, second, dst, src]


def drive_fanout(transform: Transform, model: fakemodel.FakeModel) -> object:
    fanout = transform(orchard.fanout_from_global)
    answers = []
    for n in (1, 2, 2):
        orchard.FANOUT["n"] = n
        answers.append(fanout(model, "q"))
    return answers


def drive_session(transform: Transform, model: fakemodel.FakeModel) -> object:
    chat = transform(orchard.session_chat)
    session = orchard.Session()
    session.history.append("hello")
    first = chat(model, session, "q")
    session.history.append("more context")
    return [first, chat(model, session, "q")]


def drive_shared_config(transform: Transform, model: fakemodel.FakeModel) -> object:
    shared = transform(orchard.shared_config)
    first = shared(model, "x")
    orchard.CONFIG["k"] = 2
    return [first, shared(model, "x")]


def drive_config_keywords(transform: Transform, model: fakemodel.FakeModel) -> object:
    return [
        transform(orchard.send)(model, "m", config="prod"),
        transform(orchard.send)(model, "m", config={"parallel": False}),
        transform(orchard.build)(model, "p", _config={"temperature": 0.2}),
    ]


def drive_temperature(transform: Transform, model: fakemodel.FakeModel) -> object:
    ask = transform(orchard.ask_with_temperature)
    return [ask(model, "hi"), ask(model, "hi"), ask(model, "hi", temperature=0.1)]


def answer_with_context(prompt: str) -> str:
    return f"<{prompt} served under {context.current().default_model}>"


CASES = [
    Case("four_questions", call(orchard.four_questions, "a", "b", "c", "d"), independent=4),
    Case("nested_helpers", repeat(orchard.nested_helpers, 2, "x")),
    Case("countdown", call(orchard.countdown, 3), independent=4),
    Case("persona_from_global", drive_persona),
    Case("converse", repeat(orchard.converse, 3, "hi")),
    Case("accumulate", repeat(orchard.accumulate, 4, "x")),
    Case("write_then_read", repeat(orchard.write_then_read, 2, "x")),
    Case("rebinding_callee", repeat(orchard.rebinding_callee, 2, 1)),
    Case("aliased_arguments", drive_aliasing),
    Case("caught_failure", repeat(orchard.caught_failure, 2, "x")),
    Case("method_shadows_function", repeat(orchard.method_shadows_function, 2, "x")),
    Case("jitted_method", repeat(orchard.jitted_method, 2, "ship")),
    Case("shared_config", drive_shared_config),
    Case("property_effects", repeat(orchard.property_effects, 2, 5)),
    Case("own_thread_pool", call(orchard.through_own_thread_pool, "x"), order="per_item"),
    Case("generator_calls", call(orchard.generator_calls, 3), independent=3),
    Case("dataclass_values", repeat(orchard.dataclass_values, 2, 2)),
    Case("jit_calls_jit", repeat(orchard.jit_calls_jit, 2, ["a", "b"])),
    Case("keyword_only_default", drive_temperature),
    Case("variadic_calls", call(orchard.variadic_calls, 2), independent=2),
    Case("shadowed_helper", repeat(orchard.shadowed_helper, 2, "q")),
    Case("fanout_from_global", drive_fanout),
    Case("session_object", drive_session),
    Case("pair", call(orchard.pair, "a", "b"), independent=2),
    Case("pair_failure", call(orchard.pair, "fail:a", "b"), error=ModelError),
    Case(
        "nested_fanout",
        call(orchard.nested_fanout, ["q0", "q1"]),
        independent=8,
        order="per_item",
    ),
    Case(
        "scoped_context",
        call(orchard.scoped_context, ["a", "b"]),
        order="per_item",
        answer=answer_with_context,
    ),
    Case("string_argument", call(orchard.ask_all, ["q1", "q2", "q3"], "sys"), independent=3),
    Case("config_keywords", drive_config_keywords),
    Case("tally", call(orchard.tally, ["ab", "c", "def", "gh"]), independent=4, order="per_item"),
    Case("guarded", call(orchard.guarded, ["a", "fail:b", "c"]), order="per_item"),
    Case("strict_all", call(orchard.strict_all, ["a", "fail:b", "c"]), error=ModelError),
    Case("closure_state", call(orchard.closure_state, "x"), independent=2),
    Case("relay", repeat(orchard.relay, 2, "x")),
]


def execute(case: Case, transform: Transform, latency_s: float) -> Run:
    orchard.reset()
    model = fakemodel.FakeModel(latency_s, case.answer)
    started = time.perf_counter()
    try:
        outcome: object = case.drive(transform, model)
    except Exception as error:  # The outcome under test; compared below.
        outcome = error
    wall_s = time.perf_counter() - started
    return Run(
        outcome=outcome,
        requests=model.requests(),
        reached=model.reached(),
        early=list(model.early_requests()),
        effects=effects(orchard.LOG, case.order),
        wall_s=wall_s,
    )


def effects(log: list[tuple[object, ...]], order: Literal["program", "per_item"]) -> object:
    if order == "program":
        return list(log)
    per_item: dict[str, list[tuple[object, ...]]] = collections.defaultdict(list)
    for event in log:
        per_item[repr(event[0])].append(event)
    return dict(per_item)


def divergences(case: Case, expected: Run, actual: Run) -> list[str]:
    """Every way ``actual`` departs from the plain run ``expected``."""
    found = []
    if case.error is not None:
        for run in (expected, actual):
            if type(run.outcome) is not case.error:
                found.append(f"expected {case.error.__name__}, got {run.outcome!r}")
    if isinstance(expected.outcome, Exception) or isinstance(actual.outcome, Exception):
        same_error = type(actual.outcome) is type(expected.outcome) and str(actual.outcome) == str(
            expected.outcome
        )
        if not same_error:
            found.append(f"outcome {actual.outcome!r} != {expected.outcome!r}")
    elif actual.outcome != expected.outcome:
        found.append(f"outcome {actual.outcome!r} != {expected.outcome!r}")
    if actual.requests != expected.requests:
        found.append(f"requests {dict(actual.requests)} != {dict(expected.requests)}")
    for run in (expected, actual):
        if sum(run.requests.values()) != run.reached:
            found.append(f"{run.reached} call sites but {sum(run.requests.values())} requests")
        if run.early:
            found.append(f"sent before the program reached them: {run.early}")
    if actual.effects != expected.effects:
        found.append(f"side effects {actual.effects!r} != {expected.effects!r}")
    return found


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_jit_matches_plain(case: Case) -> None:
    plain_run = execute(case, plain, SEMANTIC_LATENCY_S)
    jit_run = execute(case, xcs.jit, SEMANTIC_LATENCY_S)
    assert divergences(case, plain_run, jit_run) == []


NOT_YET_CONCURRENT = pytest.mark.xfail(
    strict=True, reason="the synchronous runtime runs one call at a time"
)
OVERLAP_CASES = [
    pytest.param(case, id=case.name, marks=NOT_YET_CONCURRENT) for case in CASES if case.independent
]


@pytest.mark.slow
@pytest.mark.parametrize("case", OVERLAP_CASES)
def test_jit_overlaps_independent_calls(case: Case) -> None:
    plain_run = execute(case, plain, OVERLAP_LATENCY_S)
    jit_run = execute(case, xcs.jit, OVERLAP_LATENCY_S)

    assert jit_run.wall_s <= OVERLAP_LATENCY_S + OVERHEAD_S, (
        f"{case.independent} independent calls of {OVERLAP_LATENCY_S}s took "
        f"{jit_run.wall_s:.3f}s under jit"
    )
    assert jit_run.wall_s <= plain_run.wall_s + NOISE_S


def test_programs_live_in_unremarkable_modules() -> None:
    for module in (orchard, fakemodel):
        assert not module.__name__.startswith(("ember", "test", "__main__")), module.__name__


def replaying(program: Program) -> Program:
    """v0.1's cache hit: a later call with equal arguments reuses the first result."""
    results: dict[str, object] = {}

    def replay(model: fakemodel.FakeModel, *args: object, **kwargs: object) -> object:
        key = repr((args, sorted(kwargs.items())))
        if key not in results:
            results[key] = program(model, *args, **kwargs)
        return results[key]

    return replay


def running_twice(program: Program) -> Program:
    """v0.1's cache hit in a user module: every recorded call ran twice."""

    def twice(*args: object, **kwargs: object) -> object:
        program(*args, **kwargs)
        return program(*args, **kwargs)

    return twice


class ExecutionError(Exception):
    """Stands in for v0.1's XCSExecutionError."""


def wrapping_errors(program: Program) -> Program:
    """v0.1 replay: a failure surfaced as a wrapper type."""

    def wrapped(*args: object, **kwargs: object) -> object:
        try:
            return program(*args, **kwargs)
        except Exception as error:
            raise ExecutionError(str(error)) from error

    return wrapped


MUTANTS = [
    (replaying, {"persona_from_global", "fanout_from_global", "session_object", "accumulate"}),
    (running_twice, {"four_questions", "countdown", "nested_fanout", "tally"}),
    (wrapping_errors, {"pair_failure", "strict_all"}),
]


@pytest.mark.parametrize(
    ("mutant", "must_catch"), MUTANTS, ids=["replaying", "running_twice", "wrapping_errors"]
)
def test_oracle_rejects_v01_behaviours(mutant: Transform, must_catch: set[str]) -> None:
    caught = set()
    for case in CASES:
        if case.name in must_catch:
            plain_run = execute(case, plain, SEMANTIC_LATENCY_S)
            mutant_run = execute(case, mutant, SEMANTIC_LATENCY_S)
            if divergences(case, plain_run, mutant_run):
                caught.add(case.name)
    assert caught == must_catch
