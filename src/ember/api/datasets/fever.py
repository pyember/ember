"""FEVER v2.0 fact verification dataset integration."""

from __future__ import annotations

from typing import Callable, Dict

from ember.api.data import DatasetInfo, HuggingFaceSource
from ember.api.datasets.catalog import get_entry
from ember.api.datasets.utils import DataTransformer

_DATASET_NAME = "fever_v2"
entry = get_entry(_DATASET_NAME)

_LABEL_MAP = {
    "SUPPORTS": "SUPPORTS",
    "REFUTES": "REFUTES",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
}


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    claim = row.get("claim") or row.get("question") or ""
    label = row.get("label") or row.get("classification") or "NOT ENOUGH INFO"
    label_text = _LABEL_MAP.get(str(label).upper(), str(label))
    evidence = row.get("evidence") or row.get("evidences") or []

    metadata = {
        "evidence": evidence,
        "license": entry.license,
    }

    return {
        "question": str(claim),
        "answer": label_text,
        "choices": {
            "A": "SUPPORTS",
            "B": "REFUTES",
            "C": "NOT ENOUGH INFO",
        },
        "metadata": metadata,
    }


def _build_source() -> DataTransformer:
    hf = entry.huggingface
    if hf is None:
        raise RuntimeError("FEVER catalog entry missing Hugging Face link")
    source = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=hf.subset,
        trust_remote_code=hf.trust_remote_code,
    )
    return DataTransformer(source, transform_fn=_normalize)


METADATA = DatasetInfo(
    name=_DATASET_NAME,
    description="FEVER v2.0 fact verification",
    size_bytes=0,
    example_count=2384,
    example_item={
        "question": "The Eiffel Tower is located in Berlin.",
        "answer": "REFUTES",
        "choices": {
            "A": "SUPPORTS",
            "B": "REFUTES",
            "C": "NOT ENOUGH INFO",
        },
        "metadata": {"license": entry.license},
    },
)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    source = _build_source()
    register_fn(_DATASET_NAME, source, METADATA)


__all__ = ["register", "METADATA"]
