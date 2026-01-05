"""MBPP (Mostly Basic Python Problems) registration helpers."""

from __future__ import annotations

from typing import Callable

from ember.api.data import DatasetInfo, HuggingFaceSource
from ember.api.datasets.catalog import DatasetCatalogEntry, get_entry
from ember.api.datasets.utils import DataTransformer

_DATASET_NAME = "mbpp"
_ENTRY: DatasetCatalogEntry = get_entry(_DATASET_NAME)

_EXAMPLE_ITEM: dict[str, object] = {
    "question": "Write a Python function that returns the square of a number.",
    "answer": "def square(x):\n    return x * x",
    "choices": {},
    "metadata": {
        "license": _ENTRY.license,
        "task_id": 0,
        "tests": ("assert square(3) == 9",),
    },
}


def _normalize(row: dict[str, object]) -> dict[str, object]:
    """Translate a raw MBPP row into Ember's dataset schema.

    Args:
        row: Dictionary returned by the MBPP Hugging Face loader.

    Returns:
        dict[str, object]: Prompt, reference solution, and metadata bundle.
    """

    metadata: dict[str, object] = {
        "license": _ENTRY.license,
        "task_id": row["task_id"],
    }
    tests = row.get("test_list")
    if isinstance(tests, (list, tuple)):
        metadata["tests"] = tuple(tests)
    challenge_tests = row.get("challenge_test_list")
    if isinstance(challenge_tests, (list, tuple)):
        metadata["challenge_tests"] = tuple(challenge_tests)
    setup = row.get("test_setup_code")
    if setup:
        metadata["test_setup_code"] = setup

    return {
        "question": str(row["text"]),
        "answer": str(row["code"]),
        "choices": {},
        "metadata": metadata,
    }


def _build_source() -> DataTransformer:
    """Return a Hugging Face backed source for MBPP.

    Returns:
        DataTransformer: Wrapped source that yields normalized MBPP rows.

    Raises:
        RuntimeError: If the MBPP catalog entry is missing its Hugging Face link.
    """
    hf = _ENTRY.huggingface
    if hf is None:
        raise RuntimeError("MBPP catalog entry missing Hugging Face link")
    base = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=hf.subset,
        trust_remote_code=hf.trust_remote_code,
    )
    return DataTransformer(base, transform_fn=_normalize)


METADATA = DatasetInfo(
    name=_DATASET_NAME,
    description=_ENTRY.description,
    size_bytes=0,
    example_count=500,
    example_item=_EXAMPLE_ITEM,
)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    """Register the MBPP dataset with the global registry.

    Args:
        register_fn: Registry helper provided by :mod:`ember.api.data`.
    """

    register_fn(_DATASET_NAME, _build_source(), METADATA)


__all__ = ["register", "METADATA"]
