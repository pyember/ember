"""Operators under the real scheduler: what overlaps and what waits.

Overlap is measured by how many requests the provider was answering at once,
which does not depend on machine speed. Waiting is measured by wall time,
only from below: every request takes `LATENCY_S`, and a sleep never ends early.
"""

import time
from collections.abc import Callable
from typing import Any

from ember import build_graph, xcs
from ember.operators import Ensemble, ModelCall

LATENCY_S = 0.25


def _timed(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_independent_model_calls_in_one_jit_call_overlap(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: model, latency_s=LATENCY_S)
    summarize, translate = ModelCall("openai/summarize"), ModelCall("anthropic/translate")

    @xcs.jit
    def both(doc: str) -> str:
        first = summarize(doc)
        second = translate(doc)
        return first.text + " " + second.text

    both("doc")
    assert fake.peak_in_flight == 2


def test_a_model_call_on_another_calls_output_waits_for_it(
    fake_models: Callable[..., Any],
) -> None:
    fake = fake_models(lambda model, prompt: model, latency_s=LATENCY_S)
    pipeline = ModelCall("openai/summarize") >> ModelCall("anthropic/translate")

    @xcs.jit
    def chained(doc: str) -> str:
        return pipeline(doc).text

    elapsed = _timed(lambda: chained("doc"))
    assert fake.peak_in_flight == 1
    assert elapsed >= 2 * LATENCY_S
    assert [call.prompt for call in fake.calls] == ["doc", "openai/summarize"]


def test_ensemble_members_overlap_outside_jit(fake_models: Callable[..., Any]) -> None:
    fake = fake_models(lambda model, prompt: model, latency_s=LATENCY_S)
    members = tuple(ModelCall(f"openai/m{index}") for index in range(4))
    Ensemble(members, quorum=4)("q")
    assert fake.peak_in_flight == 4


def test_a_graphs_e_stages_all_overlap_and_the_judge_waits(
    fake_models: Callable[..., Any],
) -> None:
    fake = fake_models(lambda model, prompt: model, latency_s=LATENCY_S)
    graph = build_graph("2E@openai/a, 3E@gemini/b, 1J@anthropic/j")
    elapsed = _timed(lambda: graph("q"))
    assert fake.peak_in_flight == 5
    assert elapsed >= 2 * LATENCY_S
    assert [call.model for call in fake.calls][-1] == "anthropic/j"
