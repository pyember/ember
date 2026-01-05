"""MathVQA dataset integration."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from ember.api.data import DatasetInfo, FileSource
from ember.api.datasets.utils import DataTransformer, resolve_dataset_path
from ember.api.exceptions import DatasetNotFoundError

_DATASET_NAME = "mathvqa"
_JSON_FILENAME = "mathvqa.jsonl"
_LICENSE = "MIT"
_MAX_IMAGE_EDGE = 5000


def _normalize(row: Dict[str, object], media_root: Path | None) -> Dict[str, object]:
    question = row.get("question") or row.get("prompt") or ""
    answer = row.get("answer") or row.get("solution") or ""
    image_rel = row.get("image") or row.get("image_path")

    media = []
    if isinstance(image_rel, str) and media_root:
        media.append({"uri": str(media_root / image_rel), "kind": "image"})

    metadata = {
        "id": row.get("id"),
        "split": row.get("split"),
        "license": _LICENSE,
    }

    return {
        "question": str(question),
        "answer": str(answer),
        "choices": {},
        "media": media,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _filter(row: Dict[str, object]) -> bool:
    width = row.get("image_width")
    height = row.get("image_height")
    try:
        # int() accepts any object at runtime; type: ignore for strict stubs
        if width and int(width) > _MAX_IMAGE_EDGE:  # type: ignore[call-overload]
            return False
        if height and int(height) > _MAX_IMAGE_EDGE:  # type: ignore[call-overload]
            return False
    except (TypeError, ValueError):
        return True
    return True


def _build_source() -> DataTransformer:
    root = resolve_dataset_path(
        _DATASET_NAME,
        env_var="EMBER_DATASET_MATHVQA_PATH",
        required_files=(_JSON_FILENAME,),
    )
    jsonl_path = root if root.is_file() else root / _JSON_FILENAME
    media_root = jsonl_path.parent if jsonl_path.is_file() else root
    images_dir = media_root / "images"
    if images_dir.exists():
        media_root = images_dir

    def transform(row: Dict[str, object]) -> Dict[str, object]:
        return _normalize(row, media_root if media_root.exists() else None)

    return DataTransformer(FileSource(jsonl_path), filter_fn=_filter, transform_fn=transform)


METADATA = DatasetInfo(
    name=_DATASET_NAME,
    description="MathVQA OCR-heavy multimodal questions",
    size_bytes=0,
    example_count=299,
    example_item={
        "question": "What is the value of 3x if x=4?",
        "answer": "12",
        "choices": {},
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
