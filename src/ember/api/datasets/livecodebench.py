"""LiveCodeBench execution task integration via Hugging Face."""

from __future__ import annotations

from typing import Callable, Dict

from ember.api.data import DatasetInfo, HuggingFaceSource
from ember.api.datasets.catalog import get_entry
from ember.api.datasets.utils import DataTransformer

_REGISTERED_NAME = "livecodebench.execution"
entry = get_entry("livecodebench")


def _normalize(row: Dict[str, object]) -> Dict[str, object]:
    program = row.get("program") or row.get("code") or ""
    input_data = row.get("input") or row.get("stdin") or ""
    output = row.get("output") or row.get("expected") or ""

    question = (
        "Given the following program, predict the output when executed with the "
        "provided input.\n" + str(program)
    )
    if input_data:
        question += f"\n\nInput:\n{input_data}"

    metadata = {
        "language": row.get("language"),
        "timeout_ms": row.get("timeout_ms"),
        "license": entry.license,
    }

    return {
        "question": question,
        "answer": str(output),
        "choices": {},
        "metadata": {k: v for k, v in metadata.items() if v is not None},
    }


def _build_source() -> DataTransformer:
    hf = entry.huggingface
    if hf is None:
        raise RuntimeError("LiveCodeBench catalog entry missing Hugging Face link")
    source = HuggingFaceSource(
        hf.repo_id,
        split=hf.split,
        config=hf.subset,
        trust_remote_code=hf.trust_remote_code,
    )
    return DataTransformer(source, transform_fn=_normalize)


METADATA = DatasetInfo(
    name=_REGISTERED_NAME,
    description="LiveCodeBench execution benchmarking tasks",
    size_bytes=0,
    example_count=479,
    example_item={
        "question": "Given the following program...",
        "answer": "42\n",
        "choices": {},
        "metadata": {"license": entry.license, "language": "python"},
    },
)


def register(register_fn: Callable[[str, DataTransformer, DatasetInfo], None]) -> None:
    source = _build_source()
    register_fn(_REGISTERED_NAME, source, METADATA)


__all__ = ["register", "METADATA"]
