"""Tests for the typed DataRecord abstractions."""

import io
import json

import pytest

from ember.api.record import (
    Choice,
    ChoiceSet,
    DataRecord,
    DatasetRef,
    MediaAsset,
    MediaBundle,
    TextContent,
)


def build_record() -> DataRecord:
    question = TextContent(text="What is 2+2?", template_id="math/basic")
    answer = TextContent(text="4", metadata={"confidence": 0.9})
    choices = ChoiceSet(
        (
            Choice(label="A", value="3"),
            Choice(label="B", value="4"),
            Choice(label="C", value="5", metadata={"hint": "too high"}),
        )
    )
    media = MediaBundle(
        (
            MediaAsset(
                uri="hf://datasets/math/images/q1.png",
                kind="image",
                sha256="deadbeef",
                format="png",
                dimensions=(512, 512),
                metadata={"role": "prompt"},
            ),
        )
    )
    return DataRecord(
        question=question,
        answer=answer,
        choices=choices,
        media=media,
        metadata={"difficulty": "easy", "tags": ["arithmetic", "demo"]},
        source=DatasetRef(name="demo", subset="math", split="validation", version="1.0"),
        record_id="demo-1",
    )


class TestDataRecordSerialization:
    """Validate JSON-safe serialization and round-trips."""

    def test_round_trip(self) -> None:
        record = build_record()
        payload = record.to_dict()

        # Ensure JSON serialization succeeds without custom hooks.
        json.dumps(payload)

        loaded = DataRecord.from_dict(payload)
        assert loaded == record
        assert loaded.metadata["difficulty"] == "easy"
        assert loaded.choices[1].value == "4"

    def test_hashable(self) -> None:
        record = build_record()
        another = build_record()
        assert record == another
        assert {record, another} == {record}


class TestChoiceSet:
    """ChoiceSet helpers are deterministic."""

    def test_from_mapping_sorts_labels(self) -> None:
        mapping = {"B": "beta", "A": "alpha"}
        choice_set = ChoiceSet.from_mapping(mapping)
        assert [choice.label for choice in choice_set] == ["A", "B"]


class TestMediaAsset:
    """Media asset helpers enforce opener semantics."""

    def test_open_without_opener(self) -> None:
        asset = MediaAsset(uri="file://x", kind="image")
        with pytest.raises(RuntimeError):
            _ = asset.open()

    def test_with_opener(self) -> None:
        asset = MediaAsset(uri="memory://blob", kind="audio")

        def _make_stream() -> io.BytesIO:
            return io.BytesIO(b"data")

        opened = asset.with_opener(_make_stream).open()
        assert opened.read() == b"data"
