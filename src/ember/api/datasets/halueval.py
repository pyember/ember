"""HaluEval hallucination classification dataset integration."""

from __future__ import annotations

from typing import Callable, Dict

from ember.api.data import DatasetInfo, HuggingFaceSource
from ember.api.datasets.catalog import get_entry
from ember.api.datasets.utils import DataTransformer

_REGISTERED_NAME = "halueval"
entry = get_entry(_REGISTERED_NAME)


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    passage = str(row.get("passage", ""))
    question = str(row.get("question", ""))
    response = str(row.get("answer", row.get("response", "")))

    prompt_sections = []
    if passage:
        prompt_sections.append(f"Passage:\n{passage}")
    if question:
        prompt_sections.append(f"Question:\n{question}")
    if response:
        prompt_sections.append(f"Response:\n{response}")
    prompt_sections.append(
        "Does the response contain hallucination with respect to the passage/context? "
        "Answer with Yes or No only."
    )
    prompt = "\n\n".join(str(section) for section in prompt_sections if section)

    raw_label = row.get("label")
    label_text = str(raw_label).strip().lower() if raw_label is not None else ""
    label_map = {
        "fail": "Yes",
        "hallucination": "Yes",
        "hallucinated": "Yes",
        "yes": "Yes",
        "1": "Yes",
        "true": "Yes",
        "pass": "No",
        "no": "No",
        "0": "No",
        "false": "No",
        "correct": "No",
        "non-hallucination": "No",
        "factual": "No",
        "y": "Yes",
        "n": "No",
    }
    answer = label_map.get(label_text)
    if answer is None:
        raise ValueError(f"Unsupported HaluEval label '{raw_label}'")

    metadata = {
        "label": raw_label,
        "source": row.get("source_ds"),
        "score": row.get("score"),
        "id": row.get("id"),
        "original_response": response,
        "license": entry.license,
    }

    return {
        "question": prompt,
        "answer": answer,
        "choices": {"A": "Yes", "B": "No"},
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _build_source() -> DataTransformer:
    hf = entry.huggingface
    if hf is None:
        raise RuntimeError("HaluEval catalog entry missing Hugging Face link")

    source = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=hf.subset,
        trust_remote_code=hf.trust_remote_code,
    )
    return DataTransformer(source, transform_fn=_normalize)


METADATA = DatasetInfo(
    name=_REGISTERED_NAME,
    description=entry.description,
    size_bytes=0,
    example_count=10_000,
    example_item={
        "question": (
            "Passage:\nArthur's Magazine (1844–1846) was an American literary periodical "
            "published in Philadelphia in the 19th century. First for Women is a woman's magazine "
            "published by Bauer Media Group in the USA.\n\n"
            "Question:\nWhich magazine was started first Arthur's Magazine or First for Women?\n\n"
            "Response:\nFirst for Women launched before Arthur's Magazine.\n\n"
            "Does the response contain hallucination with respect to the passage/context? "
            "Answer with Yes or No only."
        ),
        "answer": "Yes",
        "choices": {"A": "Yes", "B": "No"},
        "metadata": {
            "label": "FAIL",
            "source": "halueval",
            "score": 0,
            "license": entry.license,
            "original_response": "First for Women launched before Arthur's Magazine.",
        },
    },
)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    source = _build_source()
    register_fn(_REGISTERED_NAME, source, METADATA)


__all__ = ["register", "METADATA"]
