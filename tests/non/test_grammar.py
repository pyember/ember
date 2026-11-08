"""The NON grammar: one syntax, opaque model ids, typed parameters, exact columns."""

import pytest

from ember.errors import NonSyntaxError
from ember.llm.ref import ModelRef
from ember.llm.request import Sampling
from ember.non import ModelStage, Params, Role, VoteStage, parse


def _only_model_stage(spec: str) -> ModelStage:
    stage = parse(spec)[0]
    assert isinstance(stage, ModelStage)
    return stage


def test_a_full_graph_parses_into_typed_stages() -> None:
    spec = (
        "3E@openai/gpt-5(temperature=0.7, quorum=2), "
        '1V@openai/gpt-5, 1J@anthropic/x(system="be brief")'
    )
    ensemble, verifier, judge = parse(spec)
    assert ensemble == ModelStage(
        Role.ENSEMBLE,
        3,
        ModelRef("openai", "gpt-5"),
        Params(sampling=Sampling(temperature=0.7), quorum=2),
        0,
    )
    assert verifier == ModelStage(Role.VERIFIER, 1, ModelRef("openai", "gpt-5"), Params(), 44)
    assert judge == ModelStage(
        Role.JUDGE, 1, ModelRef("anthropic", "x"), Params(system="be brief"), 61
    )


@pytest.mark.parametrize(
    ("model", "provider", "name"),
    [
        ("ollama/llama3.1:8b", "ollama", "llama3.1:8b"),
        ("openai/ft:gpt-4o-mini:acme::abc123", "openai", "ft:gpt-4o-mini:acme::abc123"),
        ("vllm/meta-llama/Llama-3.1-8B", "vllm", "meta-llama/Llama-3.1-8B"),
        ("gemini/gemini-2.5-pro", "gemini", "gemini-2.5-pro"),
    ],
)
def test_model_ids_are_opaque_after_the_at_sign(model: str, provider: str, name: str) -> None:
    assert _only_model_stage(f"2E@{model}, 1M").model == ModelRef(provider, name)


def test_values_are_typed_by_the_stage_schema() -> None:
    stage = _only_model_stage(
        "2E@a/b(temperature=1e-3, top_p=1, max_output_tokens=256, seed=-4, "
        'reasoning_effort=high, system="say \\"hi\\" \\\\ bye"), 1M'
    )
    assert stage.params.sampling == Sampling(
        temperature=0.001, top_p=1.0, max_output_tokens=256, seed=-4, reasoning_effort="high"
    )
    assert stage.params.system == 'say "hi" \\ bye'
    assert isinstance(stage.params.sampling.temperature, float)


def test_quoted_strings_may_hold_delimiters() -> None:
    assert _only_model_stage('2E@a/b(system="a)b, c=d"), 1M').params.system == "a)b, c=d"


def test_whitespace_between_stages_is_free_and_m_stands_alone() -> None:
    assert parse("  2E@a/b ,1E@c/d,\n1M ") == (
        ModelStage(Role.ENSEMBLE, 2, ModelRef("a", "b"), Params(), 2),
        ModelStage(Role.ENSEMBLE, 1, ModelRef("c", "d"), Params(), 10),
        VoteStage(18),
    )


@pytest.mark.parametrize(
    ("spec", "column", "reason"),
    [
        ("", 0, "expected a stage count"),
        ("0E@a/b, 1M", 0, "a stage count is at least 1"),
        ("2.5E@a/b, 1M", 0, "whole number"),
        ("3 E@a/b, 1M", 2, "must follow its count directly"),
        ("3e@a/b, 1M", 1, "unknown stage type 'e'"),
        ("3X@a/b, 1M", 1, "unknown stage type 'X'"),
        ("3E, 1M", 2, "needs a model"),
        ("3E@, 1M", 3, "expected a model id after '@'"),
        ("3E@gpt-5, 1M", 3, "not of the form provider/name"),
        ("3E@a/b(), 1M", 7, "expected a parameter name"),
        ("3E@a/b(temp=0.7), 1M", 7, "unknown parameter 'temp'"),
        ("3E@a/b(temperature=0.1, temperature=0.9), 1M", 24, "duplicate parameter"),
        ("3E@a/b(temperature=hot), 1M", 19, "expected a number, got 'hot'"),
        ("3E@a/b(temperature=1e999), 1M", 19, "not a finite number"),
        ("3E@a/b(temperature=3), 1M", 19, "temperature must be in [0, 2]"),
        ("3E@a/b(seed=1, top_p=0), 1M", 21, "top_p must be in (0, 1]"),
        ("3E@a/b(max_output_tokens=0), 1M", 25, "max_output_tokens must be positive"),
        ("\uff13E@a/b, 1M", 0, "unexpected character"),
        ("3E@a/b(seed=\u0661), 1M", 12, "unexpected character"),
        ("3E@a/b(temperature=0.\u0665), 1M", 21, "unexpected character"),
        ("3E@a/b(seed=1.5), 1M", 12, "expected a whole number"),
        ("3E@a/b(system=7), 1M", 14, "expected a word or a quoted string"),
        ('3E@a/b(system="open), 1M', 14, "unterminated string"),
        ('3E@a/b(system="\\n"), 1M', 15, "only"),
        ("3E@a/b(quorum=0), 1M", 14, "quorum is at least 1"),
        ("3E@a/b(quorum=4), 1M", 0, "quorum 4 exceeds the stage count 3"),
        ("3E@a/b(temperature=0.5, 1M", 24, "expected a parameter name"),
        ("3E@a/b,, 1M", 7, "expected a stage count"),
        ("3E@a/b, 1M,", 11, "expected a stage count"),
        ("3E@a/b 1M", 7, "expected ',' or the end"),
        ("3E@a/b, 2M", 8, "an M stage has count 1"),
        ("3E@a/b, 1M@a/b", 10, "takes no model and no parameters"),
        ("3E@a/b, 1M(tie=raise)", 10, "takes no model and no parameters"),
        ("3E@a/b, 2V@c/d, 1M", 8, "a V stage has count 1"),
        ("3E@a/b, 1V@c/d(quorum=4), 1M", 8, "quorum 4 exceeds the 3 answers"),
        ("1J@a/b", 0, "a graph starts with an E stage"),
        ("1E@a/b, 3J@c/d, 1E@a/b, 1M", 16, "E stages come before J, V and M stages"),
        ("3E@a/b, 3J@c/d, 1V@e/f, 1M", 16, "a V stage comes directly after the E stages"),
        ("3E@a/b, 1V@c/d, 1V@c/d, 1M", 16, "a V stage comes directly after the E stages"),
        ("3E@a/b, 1M, 1J@c/d", 8, "1J and 1M leave one answer and end the graph"),
        ("3E@a/b, 1J@c/d, 1J@c/d", 8, "1J and 1M leave one answer and end the graph"),
        ("3E@a/b", 0, "a graph ends with 1J or 1M"),
        ("3E@a/b, 2J@c/d", 8, "a graph ends with 1J or 1M"),
        ("3E@a/b, 1V@c/d", 8, "a graph ends with 1J or 1M"),
        ("3E@a/b, 1M; 1J@c/d", 10, "unexpected character ';'"),
    ],
)
def test_errors_name_the_column(spec: str, column: int, reason: str) -> None:
    with pytest.raises(NonSyntaxError) as raised:
        parse(spec)
    assert (raised.value.column, raised.value.spec) == (column, spec)
    assert reason in raised.value.reason


def test_non_syntax_error_is_a_value_error_with_a_pointer() -> None:
    with pytest.raises(ValueError) as raised:
        parse("3E@a/b, 2M")
    assert str(raised.value).endswith("  3E@a/b, 2M\n          ^")
