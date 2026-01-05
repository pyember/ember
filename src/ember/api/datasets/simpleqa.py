"""SimpleQA dataset integration via the Hugging Face Hub."""

from __future__ import annotations

from typing import Callable, Dict

from ember.api.data import DatasetInfo, HuggingFaceSource
from ember.api.datasets.catalog import get_entry
from ember.api.datasets.utils import DataTransformer

_DATASET_NAME = "simpleqa"


def _normalize(row: Dict[str, str]) -> Dict[str, object]:
    question = row.get("problem") or row.get("question") or ""
    answer = row.get("answer") or row.get("response") or ""
    metadata = {
        "source_id": row.get("original_index") or row.get("id"),
        "category": row.get("topic"),
        "license": entry.license,
    }
    return {
        "question": question.strip(),
        "answer": answer.strip(),
        "choices": {},
        "metadata": {k: v for k, v in metadata.items() if v},
    }


entry = get_entry(_DATASET_NAME)


def _build_source() -> DataTransformer:
    hf = entry.huggingface
    if hf is None:
        raise RuntimeError("SimpleQA catalog entry missing Hugging Face link")
    source = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=hf.subset,
        trust_remote_code=hf.trust_remote_code,
    )
    return DataTransformer(source, transform_fn=_normalize)


METADATA = DatasetInfo(
    name=_DATASET_NAME,
    description="SimpleQA factoid questions (exact match)",
    size_bytes=0,
    example_count=4326,
    example_item={
        "question": "Example prompt?",
        "answer": "Example answer.",
        "choices": {},
        "metadata": {"license": entry.license},
    },
)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    source = _build_source()
    register_fn(_DATASET_NAME, source, METADATA)


__all__ = ["register", "METADATA"]
