"""AGIEval benchmark registration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

from ember.api.data import DatasetInfo, DataSource, HuggingFaceSource
from ember.api.datasets.catalog import DatasetCatalogEntry, get_entry
from ember.api.datasets.utils import DataTransformer

_PARENT_NAME = "agieval"
_ENTRY: DatasetCatalogEntry = get_entry(_PARENT_NAME)
_CONFIG_NAMES: tuple[str, ...] = tuple(sub.identifier for sub in _ENTRY.subtasks)

_EXAMPLE_ITEM: dict[str, object] = {
    "question": "If a student scores 92 and 88, what is the average score?",
    "answer": "90",
    "choices": {"A": "85", "B": "90", "C": "94", "D": "96"},
    "metadata": {"license": _ENTRY.license},
}


def _parse_choices(raw_choices: Iterable[object]) -> dict[str, str]:
    """Return a mapping from choice labels to text.

    Args:
        raw_choices: Ordered sequence of choice strings produced by AGIEval.

    Returns:
        Dict[str, str]: Mapping from choice label (``A``, ``B``, …) to display text.

    Raises:
        ValueError: If a choice label cannot be derived or the text is empty.
    """

    parsed: dict[str, str] = {}
    for index, choice in enumerate(raw_choices):
        text = str(choice).strip()
        label = chr(ord("A") + index)
        if text.startswith("(") and ")" in text[:5]:
            closing = text.index(")")
            candidate = text[1:closing].strip()
            if len(candidate) == 1 and candidate.isalpha():
                label = candidate.upper()
                text = text[closing + 1 :].strip()
        if not text:
            raise ValueError("Choice text cannot be empty.")
        parsed[label] = text
    return parsed


def _normalize(row: dict[str, object]) -> dict[str, object] | None:
    """Normalize a raw AGIEval row into Ember's dataset schema.

    Args:
        row: Raw dictionary from the AGIEval Hugging Face loader.

    Returns:
        dict[str, object]: Prompt, answer text, choices, and metadata.
        None if the item cannot be normalized (e.g., missing choices).
    """

    passage = row.get("passage")
    question = str(row["question"]).strip()
    prompt_parts = [
        segment for segment in (passage, question) if isinstance(segment, str) and segment.strip()
    ]
    prompt = "\n\n".join(prompt_parts)

    raw_choices = row.get("choices")

    # Handle fill-in-the-blank items (no choices) - skip them
    if raw_choices is None or not isinstance(raw_choices, (list, tuple)) or len(raw_choices) == 0:
        return None

    parsed_choices = _parse_choices(raw_choices)
    answer_key = str(row.get("answer") or "").strip().upper()

    # If answer key doesn't match any choice, skip this item
    if answer_key not in parsed_choices:
        return None

    answer_text = parsed_choices[answer_key]

    metadata: dict[str, object] = {"license": _ENTRY.license}
    explanation = row.get("descriptionAnswer")
    if isinstance(explanation, str) and explanation.strip():
        metadata["explanation"] = explanation.strip()
    other = row.get("other")
    if isinstance(other, dict):
        metadata.update(other)

    return {
        "question": prompt,
        "answer": answer_text,
        "choices": parsed_choices,
        "metadata": metadata,
    }


def _build_source(config: str) -> DataTransformer:
    """Create a Hugging Face backed source for the requested subset.

    Args:
        config: AGIEval configuration name (for example ``"lsat-ar"``).

    Returns:
        DataTransformer: Wrapped Hugging Face source that emits normalized rows.
    """

    hf = _ENTRY.huggingface
    if hf is None:
        raise RuntimeError("AGIEval catalog entry missing Hugging Face link")
    base = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=config,
        trust_remote_code=hf.trust_remote_code,
    )
    return DataTransformer(base, transform_fn=_normalize)


@dataclass(slots=True)
class _MultiConfigSource:
    """DataSource that streams each AGIEval configuration sequentially."""

    configs: tuple[str, ...]
    seed: Optional[int] = None

    def read_batches(self, batch_size: int = 32) -> Iterator[list[dict[str, object]]]:
        for config in self.configs:
            source = _build_source(config)
            if self.seed is not None:
                source = source.with_seed(self.seed)
            yield from source.read_batches(batch_size)

    def with_seed(self, seed: Optional[int]) -> "_MultiConfigSource":
        return _MultiConfigSource(self.configs, seed=seed)


PARENT_METADATA = DatasetInfo(
    name=_PARENT_NAME,
    description=_ENTRY.description,
    size_bytes=0,
    example_count=0,
    example_item=_EXAMPLE_ITEM,
)


def register(register_fn: Callable[[str, DataSource, DatasetInfo], None]) -> None:
    """Register AGIEval datasets with the global registry.

    Args:
        register_fn: Registry helper provided by :mod:`ember.api.data`.
    """

    if _CONFIG_NAMES:
        register_fn(_PARENT_NAME, _MultiConfigSource(_CONFIG_NAMES), PARENT_METADATA)

    for subtask in _ENTRY.subtasks:
        info = DatasetInfo(
            name=f"{_PARENT_NAME}.{subtask.identifier}",
            description=subtask.description,
            size_bytes=0,
            example_count=0,
            example_item=_EXAMPLE_ITEM,
        )
        register_fn(info.name, _build_source(subtask.identifier), info)


__all__ = ["register", "PARENT_METADATA"]
