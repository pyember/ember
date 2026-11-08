"""ember.data: explicit row mappings, no Hub fallback, no remote code."""

import dataclasses
import sys
from collections.abc import Iterator
from types import MappingProxyType

import pytest

from ember import data
from ember.data import DATASETS, Dataset, Example

MMLU_ROW = {
    "question": "Find the degree of Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
    "subject": "abstract_algebra",
    "choices": ["0", "4", "2", "6"],
    "answer": 1,
}
GSM8K_ROW = {
    "question": "Janet's ducks lay 16 eggs per day...",
    "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\n"
    "She makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18",
}


class FakeHub:
    """Stands in for `datasets.load_dataset`; records every call."""

    def __init__(self, rows: list[object]) -> None:
        self.rows = rows
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> Iterator[object]:
        self.calls.append((args, kwargs))
        return iter(self.rows)


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    import datasets

    fake = FakeHub([MMLU_ROW, MMLU_ROW | {"answer": 3}, MMLU_ROW | {"answer": 0}])
    monkeypatch.setattr(datasets, "load_dataset", fake)
    return fake


def test_mmlu_answer_is_the_letter_of_the_indexed_choice() -> None:
    # v0.1 kept the integer index (1) next to letter-keyed choices, so no answer matched.
    example = DATASETS["mmlu"].to_example(MMLU_ROW)
    assert example == Example(
        question=MMLU_ROW["question"],
        answer="B",
        choices=("0", "4", "2", "6"),
        metadata=MappingProxyType({"subject": "abstract_algebra"}),
    )


def test_gsm8k_answer_is_the_number_after_the_marker() -> None:
    # v0.1 used the whole worked solution as the answer.
    example = DATASETS["gsm8k"].to_example(GSM8K_ROW)
    assert example.answer == "18"
    assert example.choices == ()
    assert example.metadata["solution"].endswith("18 every day.")


@pytest.mark.parametrize(
    ("name", "row", "message"),
    [
        ("mmlu", MMLU_ROW | {"answer": 4}, "must index into the 4 choices"),
        ("mmlu", MMLU_ROW | {"answer": True}, "must index into"),
        ("mmlu", MMLU_ROW | {"choices": "0 4 2 6"}, "'choices' must be a list of strings"),
        ("gsm8k", GSM8K_ROW | {"answer": "18"}, "no '####' line"),
        ("gsm8k", {"answer": "#### 3"}, "'question' must be a string"),
    ],
)
def test_rows_that_do_not_fit_the_schema_raise(
    name: str, row: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        DATASETS[name].to_example(row)


def test_stream_opens_the_pinned_repository_without_remote_code(hub: FakeHub) -> None:
    examples = list(data.stream("mmlu"))
    assert [example.answer for example in examples] == ["B", "D", "A"]
    assert hub.calls == [
        (("cais/mmlu", "all"), {"split": "test", "streaming": True, "revision": None})
    ]


def test_stream_honours_split_and_limit(hub: FakeHub) -> None:
    examples = list(data.stream("mmlu", split="validation", limit=2))
    assert len(examples) == 2
    assert hub.calls[0][1]["split"] == "validation"


def test_unknown_name_fails_before_touching_the_hub(hub: FakeHub) -> None:
    # v0.1 sent any unknown name to the Hub, so a typo became a network lookup.
    with pytest.raises(ValueError, match="unknown dataset 'mmmlu'; known: gsm8k, mmlu"):
        data.stream("mmmlu")
    assert hub.calls == []


def test_limit_below_one_is_rejected(hub: FakeHub) -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        data.stream("mmlu", limit=0)
    assert hub.calls == []


def test_a_bad_row_raises_naming_its_position(hub: FakeHub) -> None:
    hub.rows = [MMLU_ROW, MMLU_ROW | {"answer": 9}]
    rows = data.stream("mmlu")
    assert next(rows).answer == "B"
    with pytest.raises(ValueError, match=r"cais/mmlu \(all, test\) row 1: field 'answer'"):
        next(rows)


def test_any_hub_dataset_streams_through_a_caller_mapping(hub: FakeHub) -> None:
    hub.rows = [{"prompt": "2+2?", "target": "4"}]

    def to_example(row: data.Row) -> Example:
        return Example(question=str(row["prompt"]), answer=str(row["target"]))

    spec = Dataset("someone/arith", None, "train", to_example, revision="abc123")
    assert list(data.stream(spec)) == [Example(question="2+2?", answer="4")]
    assert hub.calls == [
        (("someone/arith", None), {"split": "train", "streaming": True, "revision": "abc123"})
    ]


def test_named_datasets_can_be_narrowed_with_replace(hub: FakeHub) -> None:
    algebra = dataclasses.replace(DATASETS["mmlu"], config="abstract_algebra")
    list(data.stream(algebra, limit=1))
    assert hub.calls[0][0] == ("cais/mmlu", "abstract_algebra")


def test_missing_extra_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match=r"pip install 'ember-ai\[data\]'"):
        data.stream("gsm8k")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"question": " ", "answer": "1"}, "non-empty question"),
        ({"question": "q", "answer": ""}, "non-empty answer"),
        ({"question": "q", "answer": "A", "choices": ("only",)}, "2 to 26 choices"),
        ({"question": "q", "answer": "C", "choices": ("x", "y")}, "one of A, B, not 'C'"),
        ({"question": "q", "answer": "AB", "choices": ("x", "y")}, "one of A, B"),
    ],
)
def test_invalid_examples_cannot_be_built(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Example(**kwargs)  # type: ignore[arg-type]
