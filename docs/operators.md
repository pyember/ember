# Operators

An operator is a small, immutable object that turns one input into one output. Operators compose
model calls into larger systems: a chain of steps, an ensemble of samples, a judge that reads them,
a vote. Every example on this page runs in CI against a fake model.

## Writing an operator

An operator is a frozen, slotted dataclass with a `forward` method, typed `Operator[In, Out]`. Its
fields are its configuration, and `dataclasses.replace` is the only way to change one:

```python
import dataclasses

from ember.operators import Operator


@dataclasses.dataclass(frozen=True, slots=True)
class Exclaim(Operator[str, str]):
    mark: str = "!"

    def forward(self, input: str) -> str:
        return input + self.mark


loud = Exclaim()
louder = dataclasses.replace(loud, mark="!!!")
assert loud("hi") == "hi!"
assert louder("hi") == "hi!!!"
```

Constructing an operator whose class is not decorated with
`@dataclasses.dataclass(frozen=True, slots=True)` raises `TypeError`, so an operator cannot be
mutated after it is built. Equal configurations compare and hash equal.

## Functions and chains

`@op` turns a function of one argument into an operator when the function is defined. `>>` runs one
operator after another, and mypy checks that each step takes what the previous one returns:

```python
from ember import op


@op
def tidy(text: str) -> str:
    return " ".join(text.split())


@op
def count_words(text: str) -> int:
    return len(text.split())


pipeline = tidy >> Exclaim() >> count_words
assert pipeline("  two   words ") == 2
```

## Calling a model

`ModelCall` sends its input to a model and returns the call's `Completion`, with the text, token
usage, cost and request id. The model has no default. The input is a string, a list of `Message`,
or the `Completion` of an earlier call, whose text becomes the prompt; anything else raises
`TypeError` instead of being sent as its Python repr. A template, if given, is a `str.format`
string whose only field is `{input}`:

```python
from ember.llm import Message
from ember.operators import ModelCall

summarize = ModelCall("openai/gpt-5", template="Summarize in one sentence:\n{input}")
result = summarize("Ember composes model calls into larger systems.")
print(result.text, result.usage.output)

chat = ModelCall("anthropic/claude-sonnet-4-5", system="Answer briefly.")
print(chat([Message.user("Hi"), Message.assistant("Hello."), Message.user("Name a prime.")]).text)

refine = summarize >> ModelCall("openai/gpt-5", template="Make this shorter:\n{input}")
print(refine("Ember composes model calls into larger systems.").text)

try:
    summarize({"question": "not a prompt"})
except TypeError as error:
    print(error)
```

Outside `@xcs.jit` and `xcs.map`, the call has finished when `ModelCall` returns and a failure
raises there. Inside them the `Completion` is pending: the request is already sent, and reading any
field waits for it and raises its failure. Two `ModelCall`s that do not read each other's output
therefore have both requests sent before either is read.

## Ensembles

`Ensemble` runs its members on the same input concurrently, through `ember.xcs.map`, and returns
every member's outcome in member order: `Succeeded(member, value)` or `Failed(member, error)`.
`quorum`, which has no default, is how many members must succeed; fewer raises `QuorumError`
carrying every failure. The ensemble waits for every member before it returns.

```python
from ember.operators import Ensemble, Failed, Succeeded

panel = Ensemble(
    (
        ModelCall("openai/gpt-5"),
        ModelCall("anthropic/claude-sonnet-4-5"),
        ModelCall("gemini/gemini-2.5-pro"),
    ),
    quorum=2,
)
for outcome in panel("Name a prime number."):
    if isinstance(outcome, Succeeded):
        print(outcome.member, outcome.value.text)
    else:
        print(outcome.member, "failed:", outcome.error)
```

A member is recorded as `Failed` when its model call failed for an operational reason (rate limit,
overload, server error, transport failure, cancellation) or when it is itself an ensemble whose
quorum failed. Every other error raises at once: a bad key, a missing permission, an unknown model,
a bad request, a missing price, or a bug in your code would fail every member alike, and a quorum met
despite it would hide the outage. An ensemble cannot be empty, and its members must be operators.

## Judges, votes and verifiers

These three take `Candidates(question, answers)`.

`Judge` sends one prompt holding the question and every answer, numbered, and returns the call's
`Completion`. `MajorityVote` returns the answer most candidates agree on; you choose how answers are
compared and what happens on a tie, and neither has a default. `Verifier` checks each answer with
its own call, concurrently, and reads the verdict from the last line of the reply:

```python
from ember.operators import (
    Candidates,
    Judge,
    MajorityVote,
    TiePolicy,
    Verifier,
    fold_answer,
)

candidates = Candidates("Capital of France?", ("Paris", " paris", "Lyon"))

print(Judge(ModelCall("anthropic/claude-sonnet-4-5"))(candidates).text)

vote = MajorityVote(normalize=fold_answer, tie=TiePolicy.RAISE)(candidates)
assert (vote.answer, vote.votes) == ("Paris", 2)
assert vote.tally == (("paris", 2), ("lyon", 1))

for check in Verifier(ModelCall("openai/gpt-5"))(candidates):
    if isinstance(check, Succeeded):
        print(check.value.answer, check.value.verdict)
```

`fold_answer` collapses whitespace and case. `TiePolicy.FIRST_SEEN` lets the tied answer that
appears first win; `TiePolicy.RAISE` raises `TieError`. A verifier reply that does not end with
`VERDICT: PASS` or `VERDICT: FAIL` is `Verdict.UNREADABLE`, never a guess.

## Validation

Validation is optional and declared once, as pydantic models in the class attributes `input_model`
and `output_model`. The models must set `extra="forbid"`, which is checked when the class is
defined, and validation is strict, so nothing is coerced:

```python
import pydantic


class Question(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    text: str


@dataclasses.dataclass(frozen=True, slots=True)
class Length(Operator[Question, int]):
    input_model = Question

    def forward(self, input: Question) -> int:
        return len(input.text)


assert Length()({"text": "hello"}) == 5  # Untyped data such as parsed JSON is validated.
for bad in ({"text": 5}, {"text": "hi", "extra": 1}):
    try:
        Length()(bad)
    except pydantic.ValidationError:
        print("rejected", bad)
```

Nothing is read from parameter names or type annotations, so operators behave the same under
`from __future__ import annotations` and with types imported only under `TYPE_CHECKING`. `@op` adds
no validation; subclass `Operator` when you want it. Importing `ember.operators` loads no pydantic
until a class declares a model.

## Limits

- Operators have one synchronous entry point, `operator(input)`. There is no streaming or async
  operator API in v0.2.
- An ensemble waits for every member. Returning early once the quorum has succeeded needs
  cancellation from the xcs scheduler, which v0.2 does not have.
