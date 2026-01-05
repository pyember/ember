"""MathVista dataset integration using the published Hugging Face split."""

from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

from ember.api.data import DatasetInfo, HuggingFaceSource
from ember.api.datasets.catalog import get_entry
from ember.api.datasets.utils import DataTransformer

_DATASET_NAME = "mathvista"
entry = get_entry(_DATASET_NAME)
_MAX_WIDTH = 3000
_MAX_HEIGHT = 5000


def _filter(row: Dict[str, object]) -> bool:
    if row.get("policy_flag"):
        return False

    width = row.get("image_width") or row.get("width")
    height = row.get("image_height") or row.get("height")
    try:
        # int() accepts any object at runtime; type: ignore for strict stubs
        if width and int(width) > _MAX_WIDTH:  # type: ignore[call-overload]
            return False
        if height and int(height) > _MAX_HEIGHT:  # type: ignore[call-overload]
            return False
    except (TypeError, ValueError):
        pass
    return True


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    choices_raw = row.get("options") or row.get("choices") or []
    if isinstance(choices_raw, dict):
        choices = {str(k): str(v) for k, v in choices_raw.items()}
    else:
        options = (
            list(choices_raw)
            if isinstance(choices_raw, Sequence)
            and not isinstance(choices_raw, (str, bytes, bytearray))
            else []
        )
        choices = {chr(65 + idx): _choice_text(opt) for idx, opt in enumerate(options)}

    answer = row.get("answer") or row.get("label")
    if isinstance(answer, int) and 0 <= answer < len(choices):
        answer = list(choices.values())[answer]
    elif isinstance(answer, str) and answer in choices:
        answer = choices[answer]

    image_ref = row.get("image") or row.get("image_path") or row.get("image_id")
    media: list[dict[str, object]] = []
    if isinstance(image_ref, Mapping):
        uri = image_ref.get("path")
        if isinstance(uri, str):
            media.append({"uri": uri, "kind": "image"})
    elif isinstance(image_ref, str):
        media.append({"uri": image_ref, "kind": "image"})

    question = row.get("question") or row.get("problem") or ""

    metadata = {
        "id": row.get("id") or row.get("question_id"),
        "source": row.get("source"),
        "topic": row.get("topic"),
        "license": entry.license,
    }

    return {
        "question": str(question),
        "answer": str(answer) if answer is not None else "",
        "choices": choices,
        "media": media,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _choice_text(option: object) -> str:
    if isinstance(option, str):
        return option
    if isinstance(option, dict):
        return str(option.get("text") or option.get("value") or option)
    return str(option)


def _build_source() -> DataTransformer:
    hf = entry.huggingface
    if hf is None:
        raise RuntimeError("MathVista catalog entry missing Hugging Face link")

    source = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=hf.subset,
        trust_remote_code=hf.trust_remote_code,
    )

    def transform(row: Dict[str, object]) -> Dict[str, object]:
        return _normalize(row)

    return DataTransformer(source, filter_fn=_filter, transform_fn=transform)


METADATA = DatasetInfo(
    name=_DATASET_NAME,
    description="MathVista multimodal math reasoning (testmini subset)",
    size_bytes=0,
    example_count=991,
    example_item={
        "question": "What is the angle between AB and BC?",
        "answer": "30 degrees",
        "choices": {"A": "30", "B": "45"},
        "metadata": {"license": entry.license},
    },
)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    source = _build_source()
    register_fn(_DATASET_NAME, source, METADATA)


__all__ = ["register", "METADATA"]
