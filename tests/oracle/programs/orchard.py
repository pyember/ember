"""Ordinary programs that call a model and have side effects.

The xcs oracle runs each program plainly and under `ember.xcs.jit` and
requires the same result, the same model requests and the same side effects.
Each program reproduces a shape that broke v0.1's trace-and-replay compiler;
its docstring names the audit finding it covers. The module's name is
deliberately unremarkable: nothing in ember may treat it specially.

Every program takes the model as its first argument, so the oracle can run
the same program against the test fake now and the real client later.
"""

import dataclasses
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

from ember import context, xcs
from ember.errors import ModelError
from ember.llm import Completion

Model = Callable[..., Completion]

LOG: list[tuple[object, ...]] = []


def note(*event: object) -> None:
    """Records a side effect in the order the program performs it."""
    LOG.append(event)


def reset() -> None:
    """Restores every piece of module state the programs change."""
    LOG.clear()
    PERSONA["name"] = "pirate"
    FANOUT["n"] = 1
    CONFIG["k"] = 1


def four_questions(model: Model, a: str, b: str, c: str, d: str) -> list[str]:
    """Four independent calls written one after another (runtime #1)."""
    first = model(a)
    second = model(b)
    third = model(c)
    fourth = model(d)
    note("asked", a, b, c, d)
    return [first.text, second.text, third.text, fourth.text]


def _leaf(model: Model, x: str) -> str:
    note("leaf", x)
    return model(f"leaf {x}").text


def _middle(model: Model, x: str) -> str:
    note("middle", x)
    return _leaf(model, x) + "|"


def nested_helpers(model: Model, x: str) -> str:
    """A three-deep helper chain (compiler #4: v0.1 re-ran each frame per depth)."""
    return _middle(model, x) + _middle(model, x + "'")


def countdown(model: Model, n: int) -> list[str]:
    """Recursion with a call pending across each recursive step (compiler #4)."""
    note("countdown", n)
    if n == 0:
        return [model("liftoff").text]
    head = model(f"t-{n}")
    rest = countdown(model, n - 1)
    return [head.text, *rest]


PERSONA = {"name": "pirate"}


def persona_reply(model: Model, question: str) -> str:
    """Builds the prompt from a global (runtime #2, compiler #3: v0.1 froze it)."""
    return model(f"{PERSONA['name']}: {question}").text


def converse(model: Model, question: str) -> str:
    """Grows a list after sending it (runtime #2: v0.1 resent the grown list)."""
    messages = [question]
    reply = model(" / ".join(messages))
    messages.append(reply.text)
    note("history", len(messages))
    return reply.text


def _append_item(items: list[str], item: str) -> None:
    items.append(item)


def _snapshot(items: list[str]) -> list[str]:
    return list(items)


def accumulate(model: Model, x: str) -> list[str]:
    """A local list passed to helpers (compiler #3: v0.1 kept one list forever)."""
    items: list[str] = []
    _append_item(items, x)
    reply = model(f"sum {_snapshot(items)}")
    _append_item(items, reply.text)
    return _snapshot(items)


def _write(store: dict[str, Completion], x: str, model: Model) -> None:
    store["k"] = model(f"value {x}")


def _read(store: dict[str, Completion], model: Model) -> str:
    return model(f"read {store['k'].text}").text


def write_then_read(model: Model, x: str) -> str:
    """A writer and a reader of one dict (compiler #1: v0.1 ran them concurrently)."""
    store: dict[str, Completion] = {}
    _write(store, x, model)
    return _read(store, model)


def _bump(model: Model, n: int) -> Completion:
    n = n + 1
    return model(f"bumped {n}")


def rebinding_callee(model: Model, x: int) -> str:
    """A callee rebinds its parameter before calling (compiler #2)."""
    return _bump(model, x).text


def _push(target: list[str], value: str) -> None:
    target.append(value)


def aliased_arguments(model: Model, dst: list[str], src: list[str]) -> list[int]:
    """Writes through one of two parameters that may be the same list (compiler #3a)."""
    _push(dst, model("one").text)
    return [len(dst), len(src)]


def _risky(model: Model, x: str) -> str:
    return model(f"fail:{x}").text


def caught_failure(model: Model, x: str) -> str:
    """A failure caught by the caller, then more calls (compiler #5, #18; runtime #5)."""
    try:
        return _risky(model, x)
    except ModelError as error:
        note("fell back", error.kind.value)
        return "fallback " + model(f"after {x}").text


def run(model: Model, x: str) -> list[str]:
    """A module function with the same name as `Agent.run`."""
    note("module run", x)
    return ["module", model(x).text]


class Agent:
    """Has a method named like a module function."""

    def run(self, model: Model, x: str) -> list[str]:
        note("method run", x)
        return ["method", model(x).text]


def method_shadows_function(model: Model, x: str) -> list[str]:
    """Calls a method named like a module function (compiler #6)."""
    return Agent().run(model, x)


class Planner:
    """Has a jitted method."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    @xcs.jit
    def plan(self, model: Model, goal: str) -> list[str]:
        note("plan", self.prefix, goal)
        steps = [model(f"{self.prefix} {goal} step {i}") for i in range(2)]
        return [step.text for step in steps]


def jitted_method(model: Model, goal: str) -> list[list[str]]:
    """Calls one jitted method on two instances (compiler #6)."""
    return [Planner("a").plan(model, goal), Planner("b").plan(model, goal)]


CONFIG = {"k": 1}


def _get_config() -> dict[str, int]:
    return CONFIG


def _pick(config: dict[str, int], other: Completion) -> dict[str, int]:
    del other
    return config


def shared_config(model: Model, x: str) -> list[object]:
    """Two helpers return the same object (compiler #7)."""
    config = _get_config()
    other = model(f"other {x}")
    used = model(f"use {config['k']} {x}")
    picked = _pick(config, other)
    return [used.text, other.text, picked["k"]]


class Account:
    """A balance behind a property whose getter and setter have effects."""

    def __init__(self) -> None:
        self._balance = 10

    @property
    def balance(self) -> int:
        note("balance read")
        return self._balance

    @balance.setter
    def balance(self, value: int) -> None:
        if value < 0:
            raise ValueError("negative balance")
        note("balance set", value)
        self._balance = value


class LazyClient:
    """A property that refuses reads until something configures it."""

    def __init__(self) -> None:
        self._client: str | None = None

    @property
    def client(self) -> str:
        if self._client is None:
            raise RuntimeError("client not configured")
        return self._client

    @client.setter
    def client(self, value: str) -> None:
        self._client = value


def property_effects(model: Model, deposit: int) -> str:
    """Property getters and setters with effects (compiler #8)."""
    account = Account()
    account.balance = account.balance + deposit
    lazy = LazyClient()
    lazy.client = "configured"
    return model(f"balance {account.balance} {lazy.client}").text


def through_own_thread_pool(model: Model, x: str) -> list[str]:
    """The program's own thread pool calls the model (compiler #9)."""

    def work(i: int) -> str:
        note(("worker", i), "start")
        return model(f"{x}-{i}").text

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="orchard") as pool:
        futures = [pool.submit(work, i) for i in range(2)]
        return [future.result() for future in futures]


def _chunks(model: Model, n: int) -> Iterator[Completion]:
    for i in range(n):
        note("chunk", i)
        yield model(f"chunk {i}")


def generator_calls(model: Model, n: int) -> list[str]:
    """Calls made inside a generator the program drains (compiler p18)."""
    pending = list(_chunks(model, n))
    return [completion.text for completion in pending]


@dataclasses.dataclass
class Point:
    x: int
    y: int


def dataclass_values(model: Model, n: int) -> list[Point]:
    """Builds dataclasses from results (compiler #11)."""
    return [Point(i, len(model(f"p{i}").text)) for i in range(n)]


@xcs.jit
def inner_pair(model: Model, x: str) -> list[str]:
    """A jitted function that other programs call."""
    note("inner", x)
    return [model(x + "1").text, model(x + "2").text]


def jit_calls_jit(model: Model, xs: list[str]) -> list[list[str]]:
    """Calls a jitted function in a loop (runtime #4, compiler #8)."""
    return [inner_pair(model, x) for x in xs]


def ask_with_temperature(model: Model, question: str, *, temperature: float = 0.7) -> str:
    """A keyword-only parameter with a default (runtime #6)."""
    return model(f"{question} t={temperature}").text


def _total(*values: int) -> int:
    return sum(values)


def _options(**options: int) -> list[tuple[str, int]]:
    return sorted(options.items())


def variadic_calls(model: Model, x: int) -> list[str]:
    """Builds prompts with *args and **kwargs helpers (compiler #10)."""
    first = model(f"total {_total(x, x, x)}")
    second = model(f"options {_options(a=x)}")
    return [first.text, second.text]


def helper(model: Model, x: str) -> str:
    """A module function that a nested function shadows."""
    note("module helper", x)
    return model("module:" + x).text


def shadowed_helper(model: Model, x: str) -> str:
    """A local function shadows a module function (runtime #3)."""

    def inner(y: str) -> str:
        def helper(z: str) -> str:
            return "local:" + z

        return helper(y + "!")

    return inner(x) + " | " + helper(model, x)


FANOUT = {"n": 1}


def fanout_from_global(model: Model, question: str) -> list[str]:
    """The number of calls depends on a global (runtime p03)."""
    pending = [model(f"{question}{i}") for i in range(FANOUT["n"])]
    return [completion.text for completion in pending]


class Session:
    """Conversation state the caller mutates between calls."""

    def __init__(self) -> None:
        self.history: list[str] = []


def session_chat(model: Model, session: Session, question: str) -> str:
    """Reads an object argument that changes between calls (runtime p28)."""
    return model(" / ".join([*session.history, question])).text


def pair(model: Model, a: str, b: str) -> list[str]:
    """Two independent calls; either may fail (runtime #5)."""
    first = model(a)
    second = model(b)
    return [first.text, second.text]


def _sample(model: Model, question: str, i: int) -> str:
    note((question, i), "sample")
    return model(f"{question}/{i}").text


def _per_question(model: Model, question: str) -> list[str]:
    return xcs.map(lambda i: _sample(model, question, i), range(4))


def nested_fanout(model: Model, questions: list[str]) -> list[list[str]]:
    """A map inside a map (runtime #7: v0.1 deadlocked on its shared pool)."""
    return xcs.map(lambda question: _per_question(model, question), questions)


def _named_model(model: Model, question: str) -> str:
    return model(f"{question} via {context.current().default_model}").text


def scoped_context(model: Model, questions: list[str]) -> list[object]:
    """A context override around a map (runtime #9: v0.1 workers lost it)."""
    with context.override(default_model="fake/scoped"):
        inside = xcs.map(lambda question: _named_model(model, question), questions)
    return [inside, _named_model(model, "after")]


def ask_all(model: Model, prompts: list[str], system: str) -> list[str]:
    """One string argument shared by every item (runtime #8: v0.1 split it)."""
    return xcs.map(lambda prompt: model(prompt, system=system).text, prompts)


def send(model: Model, message: str, **options: object) -> str:
    """Accepts keyword arguments named ``config`` (runtime #17)."""
    return model(f"{message} options={sorted(options.items())}").text


def build(model: Model, prompt: str, _config: object = None) -> str:
    """Has a parameter named ``_config`` (runtime #17)."""
    return model(f"{prompt} config={_config}").text


def tally(model: Model, words: list[str]) -> list[object]:
    """Map items write to a shared dict around their calls."""
    counts: dict[str, int] = {}

    def one(word: str) -> str:
        note(word, "start")
        reply = model(f"count {word}")
        counts[word] = len(reply.text)
        note(word, "done", counts[word])
        return reply.text

    texts = xcs.map(one, words)
    return [texts, sorted(counts.items())]


def guarded(model: Model, xs: list[str]) -> list[str]:
    """Map items catch their own failures."""

    def one(x: str) -> str:
        try:
            return model(x).text
        except ModelError as error:
            note(x, "failed", error.kind.value)
            return "-"

    return xcs.map(one, xs)


def strict_all(model: Model, xs: list[str]) -> list[str]:
    """A failure escapes a map item (runtime #19)."""
    return xcs.map(lambda x: model(x).text, xs)


def fail_then_continue(model: Model, x: str) -> list[str]:
    """A failing call followed by more work: the failure must raise at one place."""
    note("before")
    first = model(f"fail:{x}")
    note("after the failing call")
    second = model(f"next {x}")
    note("after the next call")
    return [first.text, second.text]


def unread_stream(model: Model, prompt: str) -> str:
    """Starts a stream and never reads it: both runs send its request."""
    model.stream(prompt)
    note("started a stream")
    return "done"


def returned_stream(model: Model, prompt: str) -> Iterator[str]:
    """Returns a stream for the caller to read after the call returns."""
    note("returning a stream")
    return model.stream(prompt)


def streamed_answer(model: Model, prompt: str) -> str:
    """Reads a stream to the end."""
    return "".join(model.stream(prompt))


def _counter() -> Callable[[], int]:
    count = 0

    def bump() -> int:
        nonlocal count
        count += 1
        return count

    return bump


def closure_state(model: Model, x: str) -> list[object]:
    """A closure counter read at each call site (compiler #3)."""
    bump = _counter()
    first = model(f"{x} #{bump()}")
    second = model(f"{x} #{bump()}")
    return [first.text, second.text, bump()]


def relay(model: Model, x: str) -> list[str]:
    """A pending result stored under two names and read through both."""
    first = model(x)
    holder = {"a": first, "again": first}
    second = model(f"after {holder['a'].text}")
    return [holder["again"].text, second.text]
