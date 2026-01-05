"""SuperGPQA Management slice integration."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from ember.api.data import DatasetInfo, FileSource
from ember.api.datasets.utils import DataTransformer, resolve_dataset_path
from ember.api.exceptions import DatasetNotFoundError

_DATASET_NAME = "supergpqa_management"
_JSON_FILENAME = "management.jsonl"
_LICENSE = "ODC-Attribution"


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    question = row.get("question") or row.get("prompt") or ""
    choices_raw = row.get("choices") or row.get("options") or []
    answer = row.get("answer") or row.get("label")

    if isinstance(choices_raw, dict):
        choices = {str(k): str(v) for k, v in choices_raw.items()}
    else:
        options = (
            list(choices_raw)
            if isinstance(choices_raw, Sequence)
            and not isinstance(choices_raw, (str, bytes, bytearray))
            else []
        )
        choices = {chr(65 + idx): str(value) for idx, value in enumerate(options)}

    if isinstance(answer, int) and 0 <= answer < len(choices):
        answer_text = list(choices.values())[answer]
    elif isinstance(answer, str):
        answer_text = choices.get(answer, answer)
    else:
        answer_text = str(answer)

    metadata = {
        "category": row.get("category"),
        "license": _LICENSE,
    }

    return {
        "question": str(question),
        "answer": answer_text,
        "choices": choices,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _build_source() -> DataTransformer:
    root = resolve_dataset_path(
        _DATASET_NAME,
        env_var="EMBER_DATASET_SUPERGPQA_MANAGEMENT_PATH",
        required_files=(_JSON_FILENAME,),
    )
    json_path = root if root.is_file() else root / _JSON_FILENAME
    return DataTransformer(FileSource(json_path), transform_fn=_normalize)


METADATA = DatasetInfo(
    name=_DATASET_NAME,
    description="SuperGPQA management multiple-choice questions",
    size_bytes=0,
    example_count=500,
    example_item={
        "question": "What is the next step a manager should take?",
        "answer": "Consult stakeholders",
        "choices": {"A": "Consult stakeholders", "B": "Delay the project"},
        "metadata": {"license": _LICENSE},
    },
)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    try:
        source = _build_source()
    except DatasetNotFoundError:
        return
    register_fn(_DATASET_NAME, source, METADATA)


__all__ = ["register", "METADATA"]
