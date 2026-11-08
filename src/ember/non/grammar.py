"""The NON grammar: a specification string parsed into typed stages.

::

    graph  := stage ("," stage)*
    stage  := count TYPE "@" model params?      for E, J, V
            | count "M"                          majority vote, no model
    model  := provider "/" name                  name may contain ":" "." "-" "_"
    params := "(" key "=" value ("," key "=" value)* ")"
    value  := number | "quoted string" | word

The text after ``@`` is an opaque model id read up to ``(``, ``)``, ``,`` or
whitespace, so ``openai/ft:gpt-4o-mini:acme::abc123`` and
``ollama/llama3.1:8b`` arrive unchanged. Every error is a `NonSyntaxError`
carrying the column where it was detected.

Beyond the grammar, the stages come in this order: one or more E stages, at
most one V stage, then J and M stages ending in ``1J`` or ``1M``. Those two
leave a single answer, so neither appears before the end. M and V stages have
count 1.
"""

import dataclasses
import enum
import math
import re
from collections.abc import Mapping
from typing import NoReturn

from ember.errors import NonSyntaxError
from ember.llm.ref import ModelRef
from ember.llm.request import Sampling

__all__ = [
    "ModelStage",
    "Params",
    "Role",
    "Stage",
    "VoteStage",
    "parse",
]


class Role(enum.Enum):
    """What a model stage does."""

    ENSEMBLE = "E"
    JUDGE = "J"
    VERIFIER = "V"


@dataclasses.dataclass(frozen=True, slots=True)
class Params:
    """A model stage's parameters.

    Attributes:
        sampling: Generation settings for every call the stage makes.
        system: System instructions for those calls.
        quorum: For E and J, the members that must succeed (default: all);
            for V, the answers that must pass (default: 1).
    """

    sampling: Sampling = Sampling()
    system: str | None = None
    quorum: int | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ModelStage:
    """An E, J or V stage.

    Attributes:
        role: What the stage does.
        count: Calls per run for E and J; always 1 for V.
        model: The model every call uses.
        params: Typed parameters.
        column: Where the stage starts in the specification.
    """

    role: Role
    count: int
    model: ModelRef
    params: Params
    column: int


@dataclasses.dataclass(frozen=True, slots=True)
class VoteStage:
    """An M stage: a majority vote over the current answers.

    Attributes:
        column: Where the stage starts in the specification.
    """

    column: int


Stage = ModelStage | VoteStage

_PARAM_NAMES = frozenset(
    {"temperature", "top_p", "max_output_tokens", "seed", "reasoning_effort", "system", "quorum"}
)


def parse(spec: str) -> tuple[Stage, ...]:
    """Parses a NON specification.

    Example:
        >>> from ember.non import parse
        >>> [stage.column for stage in parse("3E@openai/gpt-5.5, 1J@anthropic/claude-opus-5-5")]
        [0, 19]

    Raises:
        NonSyntaxError: ``spec`` is not a valid graph.
    """
    stages = _Parser(spec, _tokenize(spec)).graph()
    _check_graph(spec, stages)
    return stages


class _Kind(enum.Enum):
    NUMBER = "a number"
    WORD = "a word"
    STRING = "a quoted string"
    MODEL = "a model id"
    AT = "'@'"
    COMMA = "','"
    LPAREN = "'('"
    RPAREN = "')'"
    EQUALS = "'='"
    END = "the end of the specification"


@dataclasses.dataclass(frozen=True, slots=True)
class _Token:
    kind: _Kind
    text: str
    column: int


_PUNCTUATION: Mapping[str, _Kind] = {
    ",": _Kind.COMMA,
    "(": _Kind.LPAREN,
    ")": _Kind.RPAREN,
    "=": _Kind.EQUALS,
}
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_INTEGER = re.compile(r"[+-]?\d+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_MODEL = re.compile(r"[^\s(),]+")


def _tokenize(spec: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(spec):
        char = spec[position]
        if char.isspace():
            position += 1
        elif char == "@":
            tokens.append(_Token(_Kind.AT, char, position))
            model = _MODEL.match(spec, position + 1)
            if model is None:
                raise NonSyntaxError(spec, position + 1, "expected a model id after '@'")
            tokens.append(_Token(_Kind.MODEL, model.group(), position + 1))
            position = model.end()
        elif char in _PUNCTUATION:
            tokens.append(_Token(_PUNCTUATION[char], char, position))
            position += 1
        elif char == '"':
            token, position = _string(spec, position)
            tokens.append(token)
        else:
            token = _word_or_number(spec, position)
            tokens.append(token)
            position += len(token.text)
    tokens.append(_Token(_Kind.END, "", len(spec)))
    return tokens


def _string(spec: str, start: int) -> tuple[_Token, int]:
    """Reads a quoted string whose only escapes are ``\\"`` and ``\\\\``."""
    chars: list[str] = []
    position = start + 1
    while position < len(spec):
        char = spec[position]
        if char == '"':
            return _Token(_Kind.STRING, "".join(chars), start), position + 1
        if char == "\\":
            escaped = spec[position + 1 : position + 2]
            if escaped not in ('"', "\\"):
                raise NonSyntaxError(spec, position, 'only \\" and \\\\ are escapes in a string')
            chars.append(escaped)
            position += 2
        else:
            chars.append(char)
            position += 1
    raise NonSyntaxError(spec, start, "unterminated string")


def _word_or_number(spec: str, position: int) -> _Token:
    number = _NUMBER.match(spec, position)
    if number is not None:
        return _Token(_Kind.NUMBER, number.group(), position)
    word = _WORD.match(spec, position)
    if word is not None:
        return _Token(_Kind.WORD, word.group(), position)
    raise NonSyntaxError(spec, position, f"unexpected character {spec[position]!r}")


class _Parser:
    """Recursive descent over the token list, one method per production."""

    def __init__(self, spec: str, tokens: list[_Token]) -> None:
        self._spec = spec
        self._tokens = tokens
        self._index = 0

    def graph(self) -> tuple[Stage, ...]:
        stages = [self._stage()]
        while self._peek().kind is _Kind.COMMA:
            self._advance()
            stages.append(self._stage())
        self._expect(_Kind.END, "expected ',' or the end of the specification")
        return tuple(stages)

    def _stage(self) -> Stage:
        count_token = self._expect(_Kind.NUMBER, "expected a stage count such as 3")
        count = self._count(count_token)
        code = self._expect(_Kind.WORD, "expected a stage type E, J, M or V")
        if code.column != count_token.column + len(count_token.text):
            self._fail(code, "the stage type must follow its count directly, as in 3E")
        if code.text == "M":
            return self._vote_stage(count_token, count)
        try:
            role = Role(code.text)
        except ValueError:
            self._fail(code, f"unknown stage type {code.text!r}; expected E, J, M or V")
        if role is Role.VERIFIER and count != 1:
            self._fail(count_token, "a V stage has count 1")
        self._expect(_Kind.AT, f"a {role.value} stage needs a model: {role.value}@provider/name")
        model = self._model(self._expect(_Kind.MODEL, "expected a model id"))
        params = self._params() if self._peek().kind is _Kind.LPAREN else Params()
        if role is not Role.VERIFIER and params.quorum is not None and params.quorum > count:
            self._fail(count_token, f"quorum {params.quorum} exceeds the stage count {count}")
        return ModelStage(role, count, model, params, count_token.column)

    def _vote_stage(self, count_token: _Token, count: int) -> VoteStage:
        if count != 1:
            self._fail(count_token, "an M stage has count 1")
        if self._peek().kind in (_Kind.AT, _Kind.LPAREN):
            self._fail(self._peek(), "an M stage takes no model and no parameters")
        return VoteStage(count_token.column)

    def _count(self, token: _Token) -> int:
        if not token.text.isdigit():
            self._fail(token, "a stage count is a whole number such as 3")
        count = int(token.text)
        if count < 1:
            self._fail(token, "a stage count is at least 1")
        return count

    def _model(self, token: _Token) -> ModelRef:
        try:
            return ModelRef.parse(token.text)
        except ValueError:
            self._fail(token, f"model {token.text!r} is not of the form provider/name")

    def _params(self) -> Params:
        opening = self._expect(_Kind.LPAREN, "expected '('")
        values: dict[str, _Token] = {}
        while True:
            key = self._expect(_Kind.WORD, "expected a parameter name")
            if key.text not in _PARAM_NAMES:
                known = ", ".join(sorted(_PARAM_NAMES))
                self._fail(key, f"unknown parameter {key.text!r}; known: {known}")
            if key.text in values:
                self._fail(key, f"duplicate parameter {key.text!r}")
            self._expect(_Kind.EQUALS, "expected '='")
            values[key.text] = self._value()
            if self._peek().kind is not _Kind.COMMA:
                break
            self._advance()
        self._expect(_Kind.RPAREN, "expected ',' or ')'")
        return self._typed_params(opening, values)

    def _value(self) -> _Token:
        token = self._advance()
        if token.kind not in (_Kind.NUMBER, _Kind.STRING, _Kind.WORD):
            self._fail(token, "expected a value")
        return token

    def _typed_params(self, opening: _Token, values: Mapping[str, _Token]) -> Params:
        temperature = self._float(values.get("temperature"))
        top_p = self._float(values.get("top_p"))
        max_output_tokens = self._int(values.get("max_output_tokens"))
        seed = self._int(values.get("seed"))
        reasoning_effort = self._str(values.get("reasoning_effort"))
        try:
            sampling = Sampling(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                seed=seed,
                reasoning_effort=reasoning_effort,
            )
        except ValueError as exc:  # A range check in Sampling; it names the field.
            self._fail(opening, str(exc))
        quorum = self._int(values.get("quorum"))
        if quorum is not None and quorum < 1:
            self._fail(values["quorum"], "quorum is at least 1")
        return Params(sampling=sampling, system=self._str(values.get("system")), quorum=quorum)

    def _float(self, token: _Token | None) -> float | None:
        if token is None:
            return None
        if token.kind is not _Kind.NUMBER:
            self._fail(token, f"expected a number, got {token.text!r}")
        value = float(token.text)
        if not math.isfinite(value):
            self._fail(token, f"{token.text} is not a finite number")
        return value

    def _int(self, token: _Token | None) -> int | None:
        if token is None:
            return None
        if token.kind is not _Kind.NUMBER or _INTEGER.fullmatch(token.text) is None:
            self._fail(token, f"expected a whole number, got {token.text!r}")
        return int(token.text)

    def _str(self, token: _Token | None) -> str | None:
        if token is None:
            return None
        if token.kind is _Kind.NUMBER:
            self._fail(token, f"expected a word or a quoted string, got {token.text}")
        return token.text

    def _peek(self) -> _Token:
        return self._tokens[self._index]

    def _advance(self) -> _Token:
        token = self._tokens[self._index]
        if token.kind is not _Kind.END:
            self._index += 1
        return token

    def _expect(self, kind: _Kind, reason: str) -> _Token:
        token = self._peek()
        if token.kind is not kind:
            self._fail(token, f"{reason}, got {token.kind.value}")
        return self._advance()

    def _fail(self, token: _Token, reason: str) -> NoReturn:
        raise NonSyntaxError(self._spec, token.column, reason)


def _check_graph(spec: str, stages: tuple[Stage, ...]) -> None:
    """Checks the stage order: E stages, an optional V, then J and M stages ending in 1J or 1M."""
    generating = 0
    while generating < len(stages) and _is_ensemble(stages[generating]):
        generating += 1
    if generating == 0:
        raise NonSyntaxError(spec, stages[0].column, "a graph starts with an E stage")
    rest = stages[generating:]
    for index, stage in enumerate(rest):
        if _is_ensemble(stage):
            raise NonSyntaxError(spec, stage.column, "E stages come before J, V and M stages")
        if isinstance(stage, ModelStage) and stage.role is Role.VERIFIER and index > 0:
            raise NonSyntaxError(spec, stage.column, "a V stage comes directly after the E stages")
        if _leaves_one_answer(stage) and index < len(rest) - 1:
            raise NonSyntaxError(spec, stage.column, "1J and 1M leave one answer and end the graph")
    if not rest or not _leaves_one_answer(rest[-1]):
        raise NonSyntaxError(spec, stages[-1].column, "a graph ends with 1J or 1M")
    if isinstance(rest[0], ModelStage) and rest[0].role is Role.VERIFIER:
        answers = sum(stage.count for stage in stages[:generating] if isinstance(stage, ModelStage))
        _check_verifier_quorum(spec, rest[0], answers)


def _leaves_one_answer(stage: Stage) -> bool:
    return isinstance(stage, VoteStage) or (stage.role is Role.JUDGE and stage.count == 1)


def _is_ensemble(stage: Stage) -> bool:
    return isinstance(stage, ModelStage) and stage.role is Role.ENSEMBLE


def _check_verifier_quorum(spec: str, stage: ModelStage, answers: int) -> None:
    quorum = stage.params.quorum
    if quorum is not None and quorum > answers:
        raise NonSyntaxError(
            spec, stage.column, f"quorum {quorum} exceeds the {answers} answers this V stage sees"
        )
