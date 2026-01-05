"""Central registry describing supported Ember datasets and subtasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class HuggingFaceLink:
    """Descriptor for loading a dataset from the Hugging Face Hub."""

    repo_id: str
    split: str
    subset: Optional[str] = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DatasetSubtask:
    """Metadata describing a child task derived from a parent corpus."""

    identifier: str
    description: str
    filters: Mapping[str, Sequence[str]]


@dataclass(frozen=True)
class DatasetCatalogEntry:
    """Declarative description for a benchmark dataset."""

    name: str
    license: str
    description: str
    huggingface: Optional[HuggingFaceLink] = None
    subtasks: Sequence[DatasetSubtask] = field(default_factory=tuple)


_CATALOG: Mapping[str, DatasetCatalogEntry] = {
    "simpleqa": DatasetCatalogEntry(
        name="simpleqa",
        license="Apache-2.0",
        description="Google SimpleQA verified evaluation set",
        huggingface=HuggingFaceLink(repo_id="google/simpleqa-verified", split="eval"),
    ),
    "fever_v2": DatasetCatalogEntry(
        name="fever_v2",
        license="CC-BY-SA-3.0",
        description="FEVER v2.0 fact verification",
        huggingface=HuggingFaceLink(
            repo_id="fever",
            split="validation",
            trust_remote_code=True,
        ),
    ),
    "mathvista": DatasetCatalogEntry(
        name="mathvista",
        license="CC-BY-SA-4.0",
        description="MathVista testmini split",
        huggingface=HuggingFaceLink(repo_id="AI4Math/MathVista", split="testmini"),
    ),
    "livecodebench": DatasetCatalogEntry(
        name="livecodebench",
        license="MIT",
        description="LiveCodeBench execution tasks",
        huggingface=HuggingFaceLink(repo_id="livecodebench/execution", split="test"),
    ),
    "halueval": DatasetCatalogEntry(
        name="halueval",
        license="CC-BY-NC-2.0",
        description="HaluEval hallucination classification benchmark",
        huggingface=HuggingFaceLink(repo_id="flowaicom/HaluEval", split="test"),
    ),
    "bbeh": DatasetCatalogEntry(
        name="bbeh",
        license="Apache-2.0",
        description="BBEH benchmark corpus",
        huggingface=HuggingFaceLink(repo_id="BBEH/bbeh", split="train"),
        subtasks=(
            DatasetSubtask(
                identifier="word_sorting",
                description="BBEH word sorting task",
                filters={"task": ("word sorting",)},
            ),
            DatasetSubtask(
                identifier="buggy_tables",
                description="BBEH buggy tables task",
                filters={"task": ("buggy tables",)},
            ),
        ),
    ),
    "mmlu_pro": DatasetCatalogEntry(
        name="mmlu_pro",
        license="Apache-2.0",
        description="MMLU-Pro evaluation set",
        huggingface=HuggingFaceLink(repo_id="TIGER-Lab/MMLU-Pro", split="test"),
        subtasks=(
            DatasetSubtask(
                identifier="health",
                description="MMLU-Pro health discipline",
                filters={"category": ("health",)},
            ),
        ),
    ),
    "agieval": DatasetCatalogEntry(
        name="agieval",
        license="MIT",
        description="AGIEval graduate-level reasoning benchmark",
        huggingface=HuggingFaceLink(
            repo_id="zacharyxxxxcr/AGIEval",
            split="validation",
            trust_remote_code=True,
        ),
        subtasks=(
            DatasetSubtask("aqua-rat", "AGIEval AQuA-RAT subset", {}),
            DatasetSubtask("gaokao-biology", "AGIEval Gaokao biology subset", {}),
            DatasetSubtask("gaokao-chemistry", "AGIEval Gaokao chemistry subset", {}),
            DatasetSubtask("gaokao-chinese", "AGIEval Gaokao Chinese subset", {}),
            DatasetSubtask("gaokao-english", "AGIEval Gaokao English subset", {}),
            DatasetSubtask("gaokao-geography", "AGIEval Gaokao geography subset", {}),
            DatasetSubtask("gaokao-history", "AGIEval Gaokao history subset", {}),
            DatasetSubtask("gaokao-mathcloze", "AGIEval Gaokao math cloze subset", {}),
            DatasetSubtask("gaokao-mathqa", "AGIEval Gaokao math QA subset", {}),
            DatasetSubtask("gaokao-physics", "AGIEval Gaokao physics subset", {}),
            DatasetSubtask("jec-qa-ca", "AGIEval JEC QA civil service subset", {}),
            DatasetSubtask("jec-qa-kd", "AGIEval JEC QA knowledge subset", {}),
            DatasetSubtask("logiqa-en", "AGIEval LogiQA English subset", {}),
            DatasetSubtask("logiqa-zh", "AGIEval LogiQA Chinese subset", {}),
            DatasetSubtask("lsat-ar", "AGIEval LSAT analytical reasoning subset", {}),
            DatasetSubtask("lsat-lr", "AGIEval LSAT logical reasoning subset", {}),
            DatasetSubtask("lsat-rc", "AGIEval LSAT reading comprehension subset", {}),
            DatasetSubtask("math", "AGIEval graduate math subset", {}),
            DatasetSubtask("sat-en", "AGIEval SAT English subset", {}),
            DatasetSubtask("sat-en-without-passage", "AGIEval SAT English (no passage) subset", {}),
            DatasetSubtask("sat-math", "AGIEval SAT math subset", {}),
        ),
    ),
    "mbpp": DatasetCatalogEntry(
        name="mbpp",
        license="CC-BY-4.0",
        description="Mostly Basic Python Problems (MBPP)",
        huggingface=HuggingFaceLink(repo_id="mbpp", split="test"),
    ),
    "table_arithmetic": DatasetCatalogEntry(
        name="table_arithmetic",
        license="Generated",
        description="Synthetic table arithmetic prompts",
    ),
    "table_bias": DatasetCatalogEntry(
        name="table_bias",
        license="Generated",
        description="Synthetic table bias prompts",
    ),
}


def get_entry(name: str) -> DatasetCatalogEntry:
    try:
        return _CATALOG[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset '{name}'") from exc


def list_catalog_entries() -> Sequence[DatasetCatalogEntry]:
    return tuple(_CATALOG.values())


__all__ = [
    "DatasetCatalogEntry",
    "DatasetSubtask",
    "HuggingFaceLink",
    "get_entry",
    "list_catalog_entries",
]
