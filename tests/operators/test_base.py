"""The Operator contract: immutability, one validation site, `op`, and `>>`."""

import dataclasses
import subprocess
import sys
import textwrap

import pydantic
import pytest

from ember import op
from ember.operators import Chain, FunctionOperator, Operator


class Req(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    n: int


class Res(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    label: str


@dataclasses.dataclass(frozen=True, slots=True)
class Double(Operator[Req, int]):
    input_model = Req

    def forward(self, input: Req) -> int:
        return input.n * 2


@dataclasses.dataclass(frozen=True, slots=True)
class Labeler(Operator[int, Res]):
    output_model = Res
    produced: object = None

    def forward(self, input: int) -> Res:
        return self.produced  # type: ignore[return-value]  # Tests feed wrong outputs.


@dataclasses.dataclass(frozen=True, slots=True)
class Suffix(Operator[str, str]):
    suffix: str = "!"

    def forward(self, input: str) -> str:
        return input + self.suffix


def test_replace_is_the_update_path_and_fields_are_frozen() -> None:
    original = Suffix()
    changed = dataclasses.replace(original, suffix="?")
    assert (original("a"), changed("a")) == ("a!", "a?")
    with pytest.raises(dataclasses.FrozenInstanceError):
        original.suffix = "?"  # type: ignore[misc]


def test_equal_configs_hash_alike_and_different_configs_differ() -> None:
    assert hash(Suffix("?")) == hash(Suffix("?"))
    assert len({Suffix(str(i)) for i in range(100)}) == 100


def test_a_subclass_that_is_not_a_frozen_slotted_dataclass_cannot_be_constructed() -> None:
    class Plain(Operator[str, str]):
        def forward(self, input: str) -> str:
            return input

    @dataclasses.dataclass(frozen=True)
    class Unslotted(Operator[str, str]):
        def forward(self, input: str) -> str:
            return input

    for cls in (Plain, Unslotted):
        with pytest.raises(TypeError, match="frozen=True, slots=True"):
            cls()


def test_a_mutable_dataclass_subclass_is_rejected_at_definition() -> None:
    with pytest.raises(TypeError, match="non-frozen"):

        @dataclasses.dataclass
        class Mutable(Operator[str, str]):
            x: int = 0


def test_input_is_validated_strictly_and_coerced_never() -> None:
    assert Double()({"n": 4}) == 8  # type: ignore[arg-type]  # Untyped data is validated.
    assert Double()(Req(n=4)) == 8
    with pytest.raises(pydantic.ValidationError):
        Double()({"n": "4"})  # type: ignore[arg-type]
    with pytest.raises(pydantic.ValidationError):
        Double()({"n": 4, "secret": 1})  # type: ignore[arg-type]


def test_output_with_an_extra_key_or_wrong_type_is_rejected() -> None:
    assert Labeler(produced=Res(label="pos"))(1) == Res(label="pos")
    with pytest.raises(pydantic.ValidationError):
        Labeler(produced={"label": "pos", "secret_cost": 0.12})(1)
    with pytest.raises(pydantic.ValidationError):
        Labeler(produced={"label": 3})(1)


def test_validation_models_must_forbid_extra_keys() -> None:
    class Lax(pydantic.BaseModel):
        n: int

    with pytest.raises(TypeError, match="extra='forbid'"):

        @dataclasses.dataclass(frozen=True, slots=True)
        class UsesLax(Operator[Lax, int]):
            input_model = Lax

    with pytest.raises(TypeError, match="pydantic model class"):

        @dataclasses.dataclass(frozen=True, slots=True)
        class UsesDict(Operator[dict[str, int], int]):
            output_model = dict  # type: ignore[assignment]


class LaxInner(pydantic.BaseModel):
    n: int


class StrictInner(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    n: int


@pytest.mark.parametrize(
    "annotation",
    [LaxInner, list[LaxInner], LaxInner | None, dict[str, tuple[int, LaxInner]]],
    ids=["direct", "list", "optional", "nested-containers"],
)
def test_nested_models_must_forbid_extra_keys_too(annotation: object) -> None:
    outer = pydantic.create_model(
        "Outer",
        __config__=pydantic.ConfigDict(extra="forbid"),
        inner=(annotation, ...),
    )
    with pytest.raises(TypeError, match="LaxInner must set"):

        @dataclasses.dataclass(frozen=True, slots=True)
        class UsesOuter(Operator[object, int]):
            input_model = outer


def test_strict_nested_models_and_self_references_are_accepted() -> None:
    class Tree(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        inner: list[StrictInner]
        children: "list[Tree]" = []

    @dataclasses.dataclass(frozen=True, slots=True)
    class UsesTree(Operator[Tree, int]):
        input_model = Tree

        def forward(self, input: Tree) -> int:
            return len(input.inner)

    assert UsesTree()({"inner": [{"n": 1}], "children": []}) == 1  # type: ignore[arg-type]
    with pytest.raises(pydantic.ValidationError):
        UsesTree()({"inner": [{"n": 1, "secret": 2}]})  # type: ignore[arg-type]


def test_an_incompletely_defined_model_fails_when_the_class_is_defined() -> None:
    class Pending(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        doc: "OnlyImportedForTypeChecking"  # type: ignore[name-defined]  # noqa: F821

    assert Pending.__pydantic_complete__ is False
    with pytest.raises(TypeError, match="Pending is not fully defined"):

        @dataclasses.dataclass(frozen=True, slots=True)
        class UsesPending(Operator[Pending, int]):
            input_model = Pending


def test_nothing_is_inferred_from_parameter_names_or_annotations() -> None:
    @dataclasses.dataclass(frozen=True, slots=True)
    class Annotated(Operator[Req, object]):
        def forward(self, input: Req) -> object:
            return input

    raw = {"n": "not a number"}
    assert Annotated()(raw) is raw  # type: ignore[arg-type]  # No input_model, no validation.


def test_op_builds_an_immutable_operator_at_decoration() -> None:
    @op
    def shout(text: str) -> str:
        return text.upper()

    assert type(shout) is FunctionOperator
    assert isinstance(shout, Operator)
    assert shout("hi") == "HI"
    with pytest.raises(dataclasses.FrozenInstanceError):
        shout.fn = str.lower  # type: ignore[misc]


@pytest.mark.parametrize("fn", [42, "upper", None])
def test_a_function_operator_needs_a_callable(fn: object) -> None:
    with pytest.raises(TypeError, match="callable"):
        FunctionOperator(fn)  # type: ignore[arg-type]


def test_op_rejects_functions_that_cannot_take_exactly_one_argument() -> None:
    def two(a: str, b: str) -> str:
        return a + b

    def none() -> str:
        return ""

    def keyword_only(*, text: str) -> str:
        return text

    for fn in (two, none, keyword_only):
        with pytest.raises(TypeError, match="one positional argument"):
            op(fn)  # type: ignore[arg-type]


def test_rshift_builds_a_flat_chain_in_order() -> None:
    first, second, third = Suffix("1"), Suffix("2"), Suffix("3")
    chained = first >> second >> third
    assert chained == Chain((first, second, third))
    assert chained("x") == "x123"
    assert (first >> (second >> third)) == chained


def test_rshift_with_a_non_operator_raises() -> None:
    with pytest.raises(TypeError):
        Suffix() >> str.upper  # type: ignore[operator]


def test_chain_checks_declared_models_of_adjacent_stages() -> None:
    @dataclasses.dataclass(frozen=True, slots=True)
    class MakesRes(Operator[int, Res]):
        output_model = Res

        def forward(self, input: int) -> Res:
            return Res(label=str(input))

    with pytest.raises(TypeError, match="produces Res but stage 1 takes Req"):
        Chain((MakesRes(), Double()))


def test_chain_rejects_lists_empty_tuples_and_plain_functions() -> None:
    with pytest.raises(TypeError, match="tuple"):
        Chain([Suffix()])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        Chain(())
    with pytest.raises(TypeError, match="@op"):
        Chain((Suffix(), str.upper))  # type: ignore[arg-type]


def test_operators_import_without_jax_equinox_or_pydantic() -> None:
    script = textwrap.dedent(
        """
        import sys
        import ember.operators, ember.non
        loaded = {"jax", "equinox", "pydantic"} & set(sys.modules)
        assert not loaded, sorted(loaded)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
