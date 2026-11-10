"""Calls each provider's real API and checks what ember reports.

Run only by the ``live-smoke`` workflow, which a maintainer starts by hand; it
spends real money (well under a dollar). The test suite never imports this file.

Usage:
    uv run python tests/live/smoke.py --providers openai,openai_chat,anthropic,gemini
"""

import argparse
import dataclasses
import os
import sys
from collections.abc import Callable

import ember
from ember import context, xcs
from ember.errors import ErrorKind, ModelError
from ember.llm import Done, Message, TextDelta, ToolResultPart, ToolSpec
from ember.llm.result import FinishReason


@dataclasses.dataclass(frozen=True)
class Target:
    """One provider and the cheap model the smoke test calls on it."""

    name: str
    model: str
    key_variable: str
    reasoning_effort: str | None
    priced: bool


TARGETS = {
    "openai": Target("openai", "openai/gpt-6-luna", "OPENAI_API_KEY", "low", True),
    # OpenAI's Chat Completions endpoint through the OpenAI-compatible provider,
    # which is how the Chat codec is reached; no catalog model uses it directly.
    "openai_chat": Target("openai_chat", "openai_chat/gpt-6-luna", "OPENAI_API_KEY", None, False),
    "anthropic": Target(
        "anthropic", "anthropic/claude-haiku-5-5", "ANTHROPIC_API_KEY", "low", True
    ),
    "gemini": Target("gemini", "gemini/gemini-3.1-flash-lite", "GEMINI_API_KEY", "low", True),
}

# Generous, because these models may reason before answering; a lower cap can
# end a reply at LENGTH, which would look like a finish-reason bug.
OUTPUT_CAP = 2000

ADD = ToolSpec(
    "add",
    "Adds two integers and returns the sum.",
    {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    },
)


class Failure(Exception):
    """A check found ember reporting something other than what it should."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


def check_plain(target: Target) -> str:
    prompt = "Reply with the single word: ok"
    answer = ember.models(target.model, prompt, max_output_tokens=OUTPUT_CAP)
    result = answer.result()
    require(result.finish_reason is FinishReason.STOP, f"finish {result.finish_reason}")
    require(bool(result.text.strip()), "empty text")
    require(result.usage.measured and result.usage.output > 0, f"usage {result.usage}")
    require(result.request_id is not None, "no request id")
    if target.priced:
        require(result.cost is not None, "priced model reported no cost")
    return f"{result.text.strip()[:20]!r} {result.usage.output} out, cost {result.cost}"


def check_stream(target: Target) -> str:
    pieces: list[str] = []
    done: Done | None = None
    events = ember.models.stream(target.model, "Count from 1 to 5.", max_output_tokens=OUTPUT_CAP)
    for event in events:
        if isinstance(event, TextDelta):
            pieces.append(event.text)
        elif isinstance(event, Done):
            done = event
    require(done is not None, "stream ended without Done")
    assert done is not None
    require(done.result.finish_reason is FinishReason.STOP, f"finish {done.result.finish_reason}")
    require("".join(pieces) == done.result.text, "deltas do not add up to the result text")
    require(done.result.usage.measured, "stream usage not measured")
    return f"{len(pieces)} deltas, {done.result.usage.output} out"


def check_tool_loop(target: Target) -> str:
    question = Message.user("What is 2 + 3? Use the add tool.")
    first = ember.models(
        target.model,
        [question],
        tools=[ADD],
        tool_choice="add",
        reasoning_effort=target.reasoning_effort,
        max_output_tokens=OUTPUT_CAP,
    ).result()
    require(len(first.tool_calls) == 1, f"tool calls {first.tool_calls}")
    call = first.tool_calls[0]
    require(call.name == "add" and call.arguments == {"a": 2, "b": 3}, f"call {call}")
    history = [question, first.message, Message("tool", (ToolResultPart(call.id, "5"),))]
    second = ember.models(
        target.model,
        history,
        tools=[ADD],
        reasoning_effort=target.reasoning_effort,
        max_output_tokens=OUTPUT_CAP,
    ).result()
    require(second.finish_reason is FinishReason.STOP, f"finish {second.finish_reason}")
    require("5" in second.text, f"answer {second.text!r}")
    return f"replayed {len(first.message.parts)} parts, answer {second.text.strip()[:30]!r}"


def check_overlap(target: Target) -> str:
    @xcs.jit
    def two(model: str) -> tuple[str, str]:
        a = ember.models(model, "Name a prime number.", max_output_tokens=OUTPUT_CAP)
        b = ember.models(model, "Name a planet.", max_output_tokens=OUTPUT_CAP)
        return a.text, b.text

    texts = two(target.model)
    graph = two.graph()
    require(graph is not None and len(graph.nodes) == 2, "expected two recorded calls")
    assert graph is not None
    require(graph.edges == (), f"independent calls recorded as dependent: {graph.edges}")
    starts = [node.start_s for node in graph.nodes]
    ends = [node.end_s for node in graph.nodes]
    require(max(starts) < min(ends), "the second call was sent after the first finished")
    return f"{len(texts)} answers, wall {graph.wall_s:.2f} s"


def check_bad_key(target: Target) -> str:
    with context.override(providers={target.name: {"api_key": "invalid-key-for-ember-smoke"}}):
        try:
            ember.models(target.model, "hi", max_output_tokens=20)
        except ModelError as error:
            require(error.kind is ErrorKind.AUTH, f"kind {error.kind}")
            require("invalid-key" not in str(error), "the key appears in the error message")
            return f"{error.kind.value}, status {error.status}"
    raise Failure("a call with an invalid key succeeded")


CHECKS: tuple[tuple[str, Callable[[Target], str]], ...] = (
    ("plain", check_plain),
    ("stream", check_stream),
    ("tool loop", check_tool_loop),
    ("overlap", check_overlap),
    ("bad key", check_bad_key),
)


def run(targets: list[Target]) -> list[tuple[str, str, bool, str]]:
    rows: list[tuple[str, str, bool, str]] = []
    for target in targets:
        for name, check in CHECKS:
            try:
                rows.append((target.name, name, True, check(target)))
            except (Failure, ModelError, ValueError) as error:
                rows.append((target.name, name, False, f"{type(error).__name__}: {error}"))
            print(("PASS" if rows[-1][2] else "FAIL"), *rows[-1][:2], rows[-1][3], flush=True)
    return rows


def write_summary(rows: list[tuple[str, str, bool, str]]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = ["| Provider | Check | Result | Detail |", "|---|---|---|---|"]
    for provider, check, passed, detail in rows:
        detail = detail.replace("|", "\\|")
        lines.append(f"| {provider} | {check} | {'pass' if passed else 'FAIL'} | {detail} |")
    with open(path, "a", encoding="utf-8") as summary:
        summary.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--providers", required=True, help="comma-separated provider names")
    names = [name.strip() for name in parser.parse_args().providers.split(",") if name.strip()]
    unknown = sorted(set(names) - set(TARGETS))
    if unknown:
        parser.error(f"unknown providers {unknown}; choose from {sorted(TARGETS)}")
    targets = [TARGETS[name] for name in names]
    missing = sorted({t.key_variable for t in targets if not os.environ.get(t.key_variable)})
    if missing:
        print(f"missing secrets: {', '.join(missing)}", file=sys.stderr)
        return 2
    providers = {
        "openai_chat": {
            "kind": "openai_compatible",
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        }
    }
    with context.override(providers=providers):
        rows = run(targets)
    write_summary(rows)
    failed = [row for row in rows if not row[2]]
    print(f"{len(rows) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
