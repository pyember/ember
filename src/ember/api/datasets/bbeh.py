"""BBEH benchmark integration with task-level subtasks."""

from __future__ import annotations

import json
from typing import Callable, Dict, Sequence

from ember.api.data import DatasetInfo, HuggingFaceSource
from ember.api.datasets.catalog import get_entry
from ember.api.datasets.utils import DataTransformer

_PARENT_NAME = "bbeh"
entry = get_entry(_PARENT_NAME)

_SUBTASK_TEMPLATE = {
    "word_sorting": DatasetInfo(
        name="bbeh.word_sorting",
        description="BBEH word sorting task",
        size_bytes=0,
        example_count=0,
        example_item={
            "question": "Sort the words alphabetically.",
            "answer": "alpha beta gamma",
            "choices": {},
            "metadata": {"license": entry.license, "task": "word sorting"},
        },
    ),
    "buggy_tables": DatasetInfo(
        name="bbeh.buggy_tables",
        description="BBEH buggy tables reconstruction task",
        size_bytes=0,
        example_count=0,
        example_item={
            "question": "Reconstruct the table and answer the query.",
            "answer": "42.0",
            "choices": {},
            "metadata": {"license": entry.license, "task": "buggy tables"},
        },
    ),
}


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    question = str(row.get("input", row.get("question", "")))
    answer_raw = row.get("target", row.get("answer"))
    if answer_raw is None:
        raise ValueError("BBEH row missing target/answer field")

    answer_text = str(answer_raw)
    task = str(row.get("task", "")).strip().lower()

    metadata: Dict[str, object] = {
        "task": row.get("task"),
        "canary": row.get("canary"),
        "mini": row.get("mini"),
        "license": entry.license,
    }

    if task == "word sorting":
        tokens = [token for token in answer_text.split() if token]
        metadata["expected_tokens"] = tuple(tokens)
    elif task == "buggy tables":
        candidate = answer_text.strip()
        if candidate.startswith("{") or candidate.startswith("["):
            payload = json.loads(candidate)
            if not isinstance(payload, dict) or "result" not in payload:
                raise ValueError("BBEH buggy tables target must be a JSON object with 'result'")
            metadata["result_payload"] = payload
        else:
            metadata["result_payload"] = {"result": answer_text}

    return {
        "question": question,
        "answer": answer_text,
        "choices": {},
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _build_source(filters: Sequence[tuple[str, Sequence[str]]] | None = None) -> DataTransformer:
    hf = entry.huggingface
    if hf is None:
        raise RuntimeError("BBEH catalog entry missing Hugging Face link")

    source = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=hf.subset,
        trust_remote_code=hf.trust_remote_code,
    )

    filter_fn: Callable[[Dict[str, object]], bool] | None = None
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
    # Register parent dataset
    parent_metadata = DatasetInfo(
        name=_PARENT_NAME,
        description=entry.description,
        size_bytes=0,
        example_count=0,
        example_item={
            "question": "Example BBEH prompt.",
            "answer": "Example answer.",
            "choices": {},
            "metadata": {"license": entry.license},
        },
    )
    register_fn(_PARENT_NAME, _build_source(), parent_metadata)

    # Register subtasks defined in the catalog
    for subtask in entry.subtasks:
        info = _SUBTASK_TEMPLATE.get(subtask.identifier)
        if info is None:
            continue
        filters = tuple(subtask.filters.items())
        register_fn(info.name, _build_source(filters=filters), info)


__all__ = ["register"]
