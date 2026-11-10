<p align="center">
  <img src="docs/assets/logo_ember_icon@2x.png" alt="Ember" width="120"/>
</p>
<p align="center">
  <img src="docs/assets/ember_workmark.svg" alt="Ember" width="280"/>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/ember-ai/"><img src="https://img.shields.io/pypi/v/ember-ai.svg" alt="PyPI"></a>
</p>

Ember composes model calls into larger systems: ensembles, judges, votes, verifiers and chains, as
small typed objects or as one line of notation. Inside `@xcs.jit`, independent model calls run at
the same time while your function still runs once, in order. Every call reports what happened:
the text, tool calls, why generation stopped, tokens by kind, the exact cost when the price is
known, and the provider's request id.

Every Python example on this page runs in CI against a fake model.

## Install

```bash
pip install ember-ai
```

Ember reads keys from `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`. To save a key
instead, run `ember configure set-key openai`, which prompts for it; keys are never command-line
arguments.

## Call a model

```python
import ember

answer = ember.models("openai/gpt-5.5", "Name three prime numbers.")
print(answer.text)
print(answer.finish_reason, answer.usage.output, answer.cost)
```

A model is named `provider/name`, or by a catalog alias such as `gpt-5.5`. The call returns a
`Completion`. Settings a model does not accept, such as `temperature` on a model that rejects it,
raise `ValueError` before anything is sent. When the model's price is unknown, `cost` is `None`
rather than a guess.

Streaming yields typed events and ends with the full result:

```python
from ember.llm import Done, TextDelta

for event in ember.models.stream("anthropic/claude-opus-5-5", "Write a haiku about rivers."):
    if isinstance(event, TextDelta):
        print(event.text, end="")
    elif isinstance(event, Done):
        print("\ntokens:", event.result.usage.output)
```

`ember.models.bind` fixes a model and its settings for reuse:

```python
terse = ember.models.bind("openai/gpt-5.5", system="Answer in one sentence.")
print(terse("Why is the sky blue?").text)
```

## Run independent calls at the same time

Inside a function decorated with `@xcs.jit`, each model call is sent as soon as it is made and
returns a pending `Completion`; reading it waits for that call only. Calls that do not read each
other's results therefore overlap, and a call that reads an earlier result waits for exactly that
result. The function still runs once, in program order, so while its calls succeed it returns the
same result, sends the same requests and has the same side effects as without `jit`. One thing
differs: a failed call raises where its result is first read (or when the function returns, if it
never is), not at the call, so code in between has already run and later calls have already been
sent.

```python
from ember import xcs


@xcs.jit
def compare(topic: str) -> str:
    pro = ember.models("openai/gpt-5.5", f"The strongest argument for {topic}.")
    con = ember.models("anthropic/claude-opus-5-5", f"The strongest argument against {topic}.")
    return ember.models(
        "openai/gpt-5.5", f"Weigh these.\nFor: {pro.text}\nAgainst: {con.text}"
    ).text


print(compare("four-day work weeks"))
graph = compare.graph()
print([node.label for node in graph.nodes], graph.edges, graph.critical_path)
```

The two arguments are requested together; the third call waits for both. `compare.graph()`
describes the last run: each call, which results it waited for, its measured timings, and the
critical path. When a provider's rate limit leaves requests waiting, the next run uses that graph
to send the request on the longest remaining path first. Nothing is cached or replayed: every run
executes your function.

`xcs.map` runs a function over items concurrently and returns the results in order:

```python
answers = xcs.map(
    lambda q: ember.models("gemini/gemini-3.1-pro-preview", q).text,
    ["Capital of Peru?", "Capital of Chile?", "Capital of Ghana?"],
)
print(answers)
```

Your code never runs in parallel with itself: functions started by `map` interleave only where one
of them waits on a result. Only network requests use threads.

## Compose with operators

An operator is a frozen dataclass that turns one input into one output. `ModelCall` calls a model,
`>>` chains steps, `Ensemble` runs several members concurrently, and `Judge`, `MajorityVote` and
`Verifier` combine their answers:

```python
from ember import op
from ember.operators import Candidates, Ensemble, Judge, MajorityVote, ModelCall, TiePolicy
from ember.operators import Succeeded, fold_answer

summarize = ModelCall("openai/gpt-5.5", template="Summarize in one sentence:\n{input}")
shorten = ModelCall("anthropic/claude-sonnet-5-5", template="Make this shorter:\n{input}")
print((summarize >> shorten)("Ember composes model calls into larger systems.").text)

panel = Ensemble(
    (
        ModelCall("openai/gpt-5.5"),
        ModelCall("anthropic/claude-opus-5-5"),
        ModelCall("gemini/gemini-3.1-pro-preview"),
    ),
    quorum=2,
)
answers = tuple(o.value.text for o in panel("Is 1009 prime?") if isinstance(o, Succeeded))
print(Judge(ModelCall("anthropic/claude-opus-5-5"))(Candidates("Is 1009 prime?", answers)).text)

vote = MajorityVote(normalize=fold_answer, tie=TiePolicy.FIRST_SEEN)
print(vote(Candidates("Capital of France?", ("Paris", " paris", "Lyon"))).answer)


@op
def word_count(text: str) -> int:
    return len(text.split())


print((summarize >> op(lambda c: c.text) >> word_count)("A long article about rivers."))
```

An ensemble returns every member's outcome, success or failure, and raises `QuorumError` when fewer
than `quorum` succeed. Rate limits, overloads and server errors count as failed members; a bad key,
an exhausted quota or an unknown model raises at once, because it would fail every member alike.
[docs/operators.md](docs/operators.md) covers validation, verifiers and the rest.

## Networks of networks in one line

```python
graph = ember.build_graph("5E@openai/gpt-5.5(reasoning_effort=low), 1J@anthropic/claude-opus-5-5")
print(graph("What is the most effective way to cut household energy use?").text)

voted = ember.build_graph("7E@gemini/gemini-3.1-pro-preview, 1M")
print(voted("Is 1009 prime? Answer yes or no.").text)
```

`E` samples, `V` verifies, `J` judges and `M` takes a majority vote. A graph is an operator, its
answer keeps every call's outcome, and its five `E` calls run at the same time.
[docs/non.md](docs/non.md) has the grammar and parameters.

## Evaluate

`ember.eval.run` runs a task over items, scores each output, and writes one JSON line per item and
per request attempt, retries included, so a run can be reconstructed from its file alone:

```python
import tempfile
from pathlib import Path

from ember.data import Example
from ember.eval import NumericAnswer, run

items = [
    Example("A box holds 12 eggs. How many eggs are in 3 boxes?", "36"),
    Example("Tom has 5 apples and eats 2. How many are left?", "3"),
]


def solve(example: Example) -> str:
    prompt = f"{example.question}\nEnd with a line 'Answer: <number>'."
    return ember.models("openai/gpt-5.5", prompt).text


records = run(
    solve,
    items,
    scorer=NumericAnswer(after="Answer:"),
    records=Path(tempfile.mkdtemp()) / "run.jsonl",
)
summary = records.summary
print(summary.mean_score, summary.failed, summary.retries, summary.unpriced_calls)
```

Failed items get no score rather than a zero, and the summary counts failures, retries and calls
whose price is unknown. `ember.data.stream("gsm8k")` and `ember.data.stream("mmlu")` read real
benchmarks from the Hugging Face Hub at pinned revisions: the first call downloads the split's file,
and later calls read it from the local cache. Install `ember-ai[data]` for them.

## Configure

`~/.ember/config.yaml`, or the file named by `EMBER_CONFIG`, holds providers and defaults. The CLI
writes it with mode 0600 (on Windows, with the permissions of its folder) and never prints a key:

```bash
ember configure set-key openai                      # prompts for the key
ember configure set local kind openai_compatible    # any OpenAI-compatible server
ember configure set local base_url http://localhost:11434/v1
ember configure show                                # keys masked
ember models                                        # the catalog: limits, prices, as-of dates
ember check openai/gpt-5.5                          # one short call: finish reason, tokens, cost
```

A provider of kind `openai_compatible` reaches any server that speaks OpenAI's Chat Completions
API with streaming, such as Ollama, vLLM, Azure OpenAI's v1 endpoint or a gateway; call its models
as `local/<model>`. Their cost is `None` unless the server reports one.

Overrides apply to a block, are never written to disk, and reach every thread ember starts for
calls made inside it:

```python
from ember import context

with context.override(providers={"openai": {"api_key": "sk-example-key-0000"}}):
    print(ember.models("openai/gpt-5.5", "Hello").text)
```

## Test your code

`FakeProvider` stands in for every provider, and `ember.llm.using` scopes a client to a block. The
fake goes through the same client as real calls, so price checks, rate limits and observers all
run:

```python
from ember.llm import Client, ModelRef, Request, using
from ember.llm.fake import FakeProvider, Reply


def respond(model: ModelRef, request: Request) -> Reply:
    return Reply(f"fake answer from {model}")


fake = FakeProvider(respond=respond)
with using(Client(provider_factory=fake.factory)):
    print(ember.models("openai/gpt-5.5", "anything").text)
print(len(fake.requests))
```

## How it is built

[docs/design/v0.2.md](docs/design/v0.2.md) is the contract the code is written to: the error
kinds, the configuration rules, the request and result types, how providers are encoded, and the
guarantees `jit` and `map` keep. An oracle suite runs programs that broke earlier versions of
`jit`, plainly and under `jit`, and requires the same results, the same requests and the same side
effects.

## Develop

```bash
git clone https://github.com/pyember/ember.git
cd ember
uv sync --group dev --all-extras
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest tests/test_errors.py    # CI runs the whole suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
