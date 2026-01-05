"""CommonGenHard concept coverage dataset integration."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from ember.api.data import DatasetInfo, FileSource
from ember.api.datasets.utils import DataTransformer, resolve_dataset_path
from ember.api.exceptions import DatasetNotFoundError

_DATASET_NAME = "commongen_hard"
_JSON_FILENAME = "commongen_hard.jsonl"
_LICENSE = "Apache-2.0"


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    concepts = row.get("concepts") or row.get("keywords") or []
    if isinstance(concepts, str):
        concept_list = [part.strip() for part in concepts.split(",") if part.strip()]
    elif isinstance(concepts, Sequence):
        concept_list = [str(item) for item in concepts]
    else:
        concept_list = []

    prompt = row.get("prompt") or row.get("question")
    if not prompt:
        prompt = "Create a paragraph using all provided concepts."

    metadata = {
        "concepts": concept_list,
        "reference": row.get("reference"),
        "license": _LICENSE,
    }

    answer = row.get("answer") or row.get("reference") or ""

    return {
        "question": str(prompt),
        "answer": str(answer),
        "choices": {},
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _build_source() -> DataTransformer:
    root = resolve_dataset_path(
        _DATASET_NAME,
        env_var="EMBER_DATASET_COMMONGEN_HARD_PATH",
        required_files=(_JSON_FILENAME,),
    )
    json_path = root if root.is_file() else root / _JSON_FILENAME
    return DataTransformer(FileSource(json_path), transform_fn=_normalize)


METADATA = DatasetInfo(
    name=_DATASET_NAME,
    description="CommonGenHard concept coverage benchmark",
    size_bytes=0,
    example_count=200,
    example_item={
        "question": "Write a paragraph that uses all listed concepts.",
        "answer": "A sample paragraph ...",
        "choices": {},
        "metadata": {"license": _LICENSE, "concepts": ["cat", "window"]},
    },
)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    try:
        source = _build_source()
    except DatasetNotFoundError:
        return
    register_fn(_DATASET_NAME, source, METADATA)


__all__ = ["register", "METADATA"]
