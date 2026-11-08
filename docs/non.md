# NON

A NON specification describes a network of model calls in one line. `build_graph` parses it and
returns an operator that runs it. Every example on this page runs in CI against a fake model.

```python
from ember import build_graph

# Five samples, then a judge that reads all five and writes the answer.
graph = build_graph("5E@openai/gpt-5(temperature=0.7), 1J@anthropic/claude-sonnet-4-5")
answer = graph("What is the most effective way to cut household energy use?")
print(answer.text)

# Seven samples and a majority vote.
voted = build_graph("7E@openai/gpt-5(temperature=0.7), 1M")
print(voted("Is 1009 prime? Answer yes or no.").text)

# Samples from two models, a verifier that keeps the answers it passes, then a judge.
checked = build_graph(
    "3E@openai/gpt-5, 3E@anthropic/claude-sonnet-4-5, 1V@openai/gpt-5, "
    "1J@anthropic/claude-sonnet-4-5"
)
print(checked("Why is the sky blue? Two sentences.").text)
```

## Stages

| Stage | What it does |
|-------|--------------|
| `nE@model` | `n` calls to the model with the question. All E stages run concurrently, and their answers are pooled in stage order. |
| `1V@model` | Checks each answer with its own call and keeps the ones that pass. |
| `nJ@model` | `n` judges, each reading the question and every answer. Their answers replace the answers. |
| `1M` | Majority vote: the answer most calls agree on, compared ignoring case and whitespace. The tied answer seen first wins a tie. |

Stages come in this order: one or more E stages, at most one V stage, then J and M stages ending in
`1J` or `1M`. Those two leave a single answer, so neither appears before the end. V and M have
count 1.

Models are written `provider/name`, and everything after `@` up to `(`, `)`, `,` or a space is the
model id, untouched: `ollama/llama3.1:8b` and `openai/ft:gpt-4o-mini:acme::abc123` work.

## Parameters

E, J and V stages take parameters in parentheses:

| Parameter | Type | Meaning |
|-----------|------|---------|
| `temperature`, `top_p` | number | Sampling settings, range-checked. |
| `max_output_tokens`, `seed` | whole number | Per call. |
| `reasoning_effort` | word | For models that support it. |
| `system` | word or `"quoted string"` | System instructions. Escapes are `\"` and `\\`. |
| `quorum` | whole number | E and J: calls that must succeed (default all). V: answers that must pass (default 1). |

```python
tolerant = build_graph(
    '5E@openai/gpt-5(temperature=0.9, quorum=3, system="Answer in one word."), 1M'
)
print(tolerant("Largest planet?").text)
```

A stage whose successes fall short of its quorum raises `QuorumError`. Rate limits, overload,
server and transport errors count as failed calls; any other error, such as a bad key or an unknown
model, raises at once.

## What a run returns

A graph is an `Operator[str, Answer]`. `Answer.text` is the answer, and `Answer.stages` keeps one
record per stage with every call's outcome, failures included:

```python
from ember.non import Sampled
from ember.operators import Failed

for record in answer.stages:
    if isinstance(record, Sampled):
        failed = sum(isinstance(outcome, Failed) for outcome in record.outcomes)
        print(record.stage.role, len(record.outcomes), "calls,", failed, "failed")
```

## Mistakes

A mistake raises `NonSyntaxError`, a `ValueError` that names the column:

```python
from ember.errors import NonSyntaxError

for spec in ("0E@openai/gpt-5, 1M", "3E@openai/gpt-5(temperature=hot), 1M", "3E@gpt-5, 1M"):
    try:
        build_graph(spec)
    except NonSyntaxError as error:
        print(error.column, error.reason)
```

Counts below 1, unknown or duplicate parameters, values of the wrong type, model ids without a
provider and stages out of order are all rejected before anything is sent.

## Planning

`graph.plan(resolve)` bounds one run before it starts, given a function that returns a model's
catalog entry. Per stage and in total, it gives the most model calls, output tokens and output cost
a run can incur, from each stage's `max_output_tokens` and the catalog's limits and prices. Every
figure is an upper bound. Input tokens are not bounded, because they depend on the question and on
what earlier stages write, and calls the client retries are not counted. A bound is `None` when the
catalog does not know the model's limit or price.
