"""ember.data: explicit row mappings, pinned files, no remote code, no network."""

import dataclasses
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import MappingProxyType, ModuleType

import pyarrow
import pyarrow.parquet
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


# Hub commits read from https://huggingface.co/api/datasets/<id> on 2026-11-07.
MMLU_SHA = "c30699e8356da336a370243923dbaf21066bb9fe"
GSM8K_SHA = "740312add88f781978c0658806c59bc2815b9866"
ONE_FILE_PER_SPLIT = "{config}/{split}-00000-of-00001.parquet"


def arith_example(row: data.Row) -> Example:
    return Example(question=str(row["prompt"]), answer=str(row["target"]))


# A dataset with no parquet_file, so `stream` reads it through `datasets`.
STREAMED = Dataset("someone/arith", None, "train", arith_example, revision="abc123")


def reached_the_hub(*args: object, **kwargs: object) -> None:
    raise AssertionError(f"a test reached the Hub: {args} {kwargs}")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fails any test that would download or stream from the Hub without a fake."""
    import datasets
    import huggingface_hub

    monkeypatch.setattr(datasets, "load_dataset", reached_the_hub)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", reached_the_hub)


class FakeDownloads:
    """Stands in for `huggingface_hub.hf_hub_download`; records every call.

    Each call writes ``rows`` to a new Parquet file and returns its path, as
    the real function returns the path of the file in its cache. It takes
    keyword arguments only, so a call that relies on argument order fails.
    """

    def __init__(self, directory: Path, rows: list[dict[str, object]]) -> None:
        self.directory = directory
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        path = self.directory / f"download-{len(self.calls)}.parquet"
        pyarrow.parquet.write_table(pyarrow.Table.from_pylist(self.rows), path)
        return str(path)


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDownloads:
    import huggingface_hub

    fake = FakeDownloads(tmp_path, [MMLU_ROW, MMLU_ROW | {"answer": 3}, MMLU_ROW | {"answer": 0}])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return fake


def downloaded(repo: str, filename: str, revision: str) -> dict[str, object]:
    return {"repo_id": repo, "filename": filename, "repo_type": "dataset", "revision": revision}


class Absent(importlib.abc.MetaPathFinder):
    """Makes one top-level package fail to import, as when it is not installed."""

    def __init__(self, package: str) -> None:
        self.package = package

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == self.package:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def make_absent(monkeypatch: pytest.MonkeyPatch, package: str) -> None:
    for name in [name for name in sys.modules if name.partition(".")[0] == package]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "meta_path", [Absent(package), *sys.meta_path])


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


def test_named_datasets_read_the_pinned_file_and_never_stream(
    files: FakeDownloads, hub: FakeHub
) -> None:
    examples = list(data.stream("mmlu"))
    assert [example.answer for example in examples] == ["B", "D", "A"]
    assert files.calls == [downloaded("cais/mmlu", "all/test-00000-of-00001.parquet", MMLU_SHA)]
    assert hub.calls == []


def test_gsm8k_reads_its_pinned_file(files: FakeDownloads) -> None:
    files.rows = [GSM8K_ROW]
    assert [example.answer for example in data.stream("gsm8k")] == ["18"]
    assert files.calls == [
        downloaded("openai/gsm8k", "main/test-00000-of-00001.parquet", GSM8K_SHA)
    ]


def test_stream_honours_split_and_limit(files: FakeDownloads) -> None:
    examples = list(data.stream("mmlu", split="validation", limit=2))
    assert len(examples) == 2
    assert files.calls[0]["filename"] == "all/validation-00000-of-00001.parquet"


def test_hub_errors_pass_through_when_the_stream_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    class EntryMissing(Exception):
        pass

    def missing(**kwargs: object) -> str:
        raise EntryMissing(f"no file {kwargs['filename']}")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(EntryMissing, match="no file all/nope-00000-of-00001.parquet"):
        data.stream("mmlu", split="nope")


def test_unknown_name_fails_before_touching_the_hub() -> None:
    # v0.1 sent any unknown name to the Hub, so a typo became a network lookup.
    with pytest.raises(ValueError, match="unknown dataset 'mmmlu'; known: gsm8k, mmlu"):
        data.stream("mmmlu")


def test_limit_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        data.stream("mmlu", limit=0)


def test_a_bad_row_raises_naming_its_position(files: FakeDownloads) -> None:
    files.rows = [MMLU_ROW, MMLU_ROW | {"answer": 9}]
    rows = data.stream("mmlu")
    assert next(rows).answer == "B"
    with pytest.raises(ValueError, match=r"cais/mmlu \(all, test\) row 1: field 'answer'"):
        next(rows)


def test_any_hub_dataset_streams_through_a_caller_mapping(hub: FakeHub) -> None:
    hub.rows = [{"prompt": "2+2?", "target": "4"}]
    assert list(data.stream(STREAMED)) == [Example(question="2+2?", answer="4")]
    assert hub.calls == [
        (("someone/arith", None), {"split": "train", "streaming": True, "revision": "abc123"})
    ]


def test_any_hub_dataset_with_a_parquet_file_is_read_from_it(files: FakeDownloads) -> None:
    files.rows = [{"prompt": "2+2?", "target": "4"}]
    spec = dataclasses.replace(STREAMED, config="main", parquet_file="{config}/{split}.parquet")
    assert list(data.stream(spec)) == [Example(question="2+2?", answer="4")]
    assert files.calls == [downloaded("someone/arith", "main/train.parquet", "abc123")]


def test_named_datasets_can_be_narrowed_with_replace(files: FakeDownloads) -> None:
    algebra = dataclasses.replace(DATASETS["mmlu"], config="abstract_algebra")
    list(data.stream(algebra, limit=1))
    assert files.calls == [
        downloaded("cais/mmlu", "abstract_algebra/test-00000-of-00001.parquet", MMLU_SHA)
    ]


@pytest.mark.parametrize(
    ("dataset", "package"),
    [("gsm8k", "huggingface_hub"), ("gsm8k", "pyarrow"), (STREAMED, "datasets")],
)
def test_missing_extra_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch, dataset: str | Dataset, package: str
) -> None:
    # A download started before the missing import would fail the test with AssertionError.
    make_absent(monkeypatch, package)
    with pytest.raises(ImportError, match=r"pip install 'ember-ai\[data\]'") as raised:
        data.stream(dataset)
    assert raised.value.name == package


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


def test_named_datasets_are_pinned_to_a_hub_commit_and_file() -> None:
    # The design promised pinned datasets; v0.2's first cut read the moving default branch.
    assert DATASETS["mmlu"].revision == MMLU_SHA
    assert DATASETS["gsm8k"].revision == GSM8K_SHA
    assert {spec.parquet_file for spec in DATASETS.values()} == {ONE_FILE_PER_SPLIT}


def test_a_stream_says_where_its_examples_come_from(files: FakeDownloads) -> None:
    examples = data.stream("mmlu", split="dev", limit=2)
    assert examples.source == data.Source("cais/mmlu", "all", "dev", MMLU_SHA, 2)
    assert len(list(examples)) == 2


def test_an_empty_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="split must be a non-empty string"):
        data.stream("mmlu", split="")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"path": ""}, "path must be a non-empty string"),
        ({"split": ""}, "split must be a non-empty string"),
        ({"config": ""}, "config must be None or a non-empty string"),
        ({"revision": ""}, "revision must be None or a non-empty string"),
        ({"parquet_file": ""}, "parquet_file must be None or a non-empty string"),
        ({"parquet_file": "data/train.parquet"}, "parquet_file must contain {split}"),
        ({"parquet_file": "{name}/{split}.parquet"}, "may use only {config} and {split}"),
        ({"parquet_file": "{config}/{split}.parquet"}, "uses {config}, but config is None"),
        ({"parquet_file": "{split"}, "not a valid template"),
    ],
)
def test_invalid_datasets_cannot_be_built(kwargs: dict[str, str], message: str) -> None:
    fields = {"path": "a/b", "config": None, "split": "train", "revision": None} | kwargs
    with pytest.raises(ValueError, match=message):
        Dataset(to_example=DATASETS["gsm8k"].to_example, **fields)  # type: ignore[arg-type]


def test_a_broken_install_is_not_reported_as_a_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "datasets"
    broken.mkdir()
    (broken / "__init__.py").write_text("import ember_test_missing_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "datasets")
    with pytest.raises(ImportError) as raised:
        data.stream(STREAMED)
    assert raised.value.name == "ember_test_missing_dependency"
    assert "ember-ai[data]" not in str(raised.value)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"question": "q", "answer": 4}, "answer must be a string"),
        ({"question": None, "answer": "4"}, "question must be a string"),
        ({"question": "q", "answer": "A", "choices": ["x", "y"]}, "choices must be a tuple"),
        ({"question": "q", "answer": "A", "choices": ("x", 2)}, "choices must be a tuple"),
        ({"question": "q", "answer": "4", "metadata": {"n": 1}}, "metadata must map strings"),
    ],
)
def test_wrongly_typed_examples_raise_value_error(kwargs: dict[str, object], message: str) -> None:
    # v0.1-style rows with an integer answer raised AttributeError, which lost the row position.
    with pytest.raises(ValueError, match=message):
        Example(**kwargs)  # type: ignore[arg-type]


def test_examples_own_their_metadata_and_hash() -> None:
    metadata = {"subject": "algebra"}
    example = Example(question="q", answer="4", metadata=metadata)
    metadata["subject"] = "changed"
    assert example.metadata == {"subject": "algebra"}
    assert hash(example) == hash(Example(question="q", answer="4"))


def test_a_mapping_that_builds_a_bad_example_names_the_row(hub: FakeHub) -> None:
    hub.rows = [{"q": "2+2?", "a": 4}]
    spec = Dataset("someone/arith", None, "train", lambda row: Example(row["q"], row["a"]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"someone/arith \(default, train\) row 0: answer must"):
        list(data.stream(spec))
