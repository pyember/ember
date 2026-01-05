"""MMLU-Pro dataset integration with discipline-level subtasks."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

from ember.api.data import DatasetInfo, HuggingFaceSource
from ember.api.datasets.catalog import get_entry
from ember.api.datasets.utils import DataTransformer

_PARENT_NAME = "mmlu_pro"
entry = get_entry(_PARENT_NAME)

_SUBTASK_TEMPLATES = {
    "health": DatasetInfo(
        name="mmlu_pro.health",
        description="MMLU-Pro health discipline",
        size_bytes=0,
        example_count=0,
        example_item={
            "question": "Which symptom is associated with condition X?",
            "answer": "Fever",
            "choices": {"A": "Fever", "B": "Fatigue"},
            "metadata": {"license": entry.license, "category": "health"},
        },
    ),
}


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    question = row.get("question") or ""
    raw_options = row.get("options")
    options: Sequence[object] = (
        list(raw_options) if isinstance(raw_options, (list, tuple)) else []
    )
    answer = row.get("answer")

    choices: Dict[str, str]
    if options:
        choices = {chr(65 + idx): str(value) for idx, value in enumerate(options)}
    else:
        choices = {}

    answer_letter: str | None = None
    answer_value: object | None = None
    if isinstance(answer, int):
        if 0 <= answer < len(options):
            answer_letter = chr(65 + answer)
            answer_value = options[answer]
        else:
            answer_letter = str(answer)
    elif isinstance(answer, str):
        normalized = answer.strip().upper()
        if len(normalized) == 1 and "A" <= normalized <= "Z":
            idx = ord(normalized) - 65
            answer_letter = normalized
            if 0 <= idx < len(options):
                answer_value = options[idx]
        elif options:
            for idx, option in enumerate(options):
                if str(option).strip().lower() == answer.strip().lower():
                    answer_letter = chr(65 + idx)
                    answer_value = option
                    break
        else:
            answer_letter = normalized
    else:
        answer_letter = str(answer) if answer is not None else None

    if answer_value is None and answer_letter and answer_letter in choices:
        answer_value = choices[answer_letter]

    metadata = {
        "category": row.get("category"),
        "question_id": row.get("question_id"),
        "source": row.get("src"),
        "license": entry.license,
        "answer_text": str(answer_value) if answer_value is not None else None,
    }

    return {
        "question": str(question),
        "answer": str(answer_letter) if answer_letter is not None else "",
        "choices": choices,
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _build_source(filters: Sequence[tuple[str, Sequence[str]]] | None = None) -> DataTransformer:
    hf = entry.huggingface
    if hf is None:
        raise RuntimeError("MMLU-Pro catalog entry missing Hugging Face link")

    source = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=hf.subset,
        trust_remote_code=hf.trust_remote_code,
    )

    filter_fn = None
    if filters:
        filter_pairs = [(key, set(values)) for key, values in filters]

        def predicate(row: Dict[str, object]) -> bool:
            for key, allowed in filter_pairs:
                if row.get(key) not in allowed:
                    return False
            return True

        filter_fn = predicate

    return DataTransformer(source, filter_fn=filter_fn, transform_fn=_normalize)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    parent_metadata = DatasetInfo(
        name=_PARENT_NAME,
        description=entry.description,
        size_bytes=0,
        example_count=0,
        example_item={
            "question": "Example MMLU-Pro prompt.",
            "answer": "Example answer.",
            "choices": {"A": "Option A", "B": "Option B"},
            "metadata": {"license": entry.license},
        },
    )
    register_fn(_PARENT_NAME, _build_source(), parent_metadata)

    for subtask in entry.subtasks:
        info = _SUBTASK_TEMPLATES.get(subtask.identifier)
        if info is None:
            continue
        filters = tuple(subtask.filters.items())
        register_fn(info.name, _build_source(filters=filters), info)


__all__ = ["register"]
