from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from ember.api.data import list_datasets, register, stream
from ember.api.datasets import (
    agieval,
    bbeh,
    fever,
    halueval,
    livecodebench,
    mathvista,
    mbpp,
    mmlu_pro,
    simpleqa,
)
from ember.api.datasets.table_tasks import register as register_table_tasks


@dataclass
class StubSource:
    rows: List[Dict[str, Any]]

    def read_batches(self, batch_size: int = 32):
        yield self.rows

    def with_seed(self, seed):  # pragma: no cover
        return self


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    # Ensure each test starts with a clean registry snapshot by re-instantiating
    # the module registry.
    from ember.api import data as data_module

    original_registry = data_module._registry
    data_module._registry = data_module._RegistryClass()
    try:
        yield
    finally:
        data_module._registry = original_registry


def test_simpleqa_register(monkeypatch):
    rows = [
        {
            "question": "Who wrote Nineteen Eighty-Four?",
            "answer": "George Orwell",
            "original_index": 1,
            "topic": "Literature",
        }
    ]
    monkeypatch.setattr(simpleqa, "HuggingFaceSource", lambda *args, **kwargs: StubSource(rows))

    simpleqa.register(register)

    assert "simpleqa" in list_datasets()
    record = stream("simpleqa").first(1)[0]
    assert record.question.text == "Who wrote Nineteen Eighty-Four?"
    assert record.answer.text == "George Orwell"
    assert record.metadata["license"] == "Apache-2.0"


def test_fever_register(monkeypatch):
    rows = [
        {
            "claim": "The sky is blue.",
            "label": "SUPPORTS",
            "evidence": [["Sky", 0, "The sky is blue."]],
        }
    ]
    monkeypatch.setattr(fever, "HuggingFaceSource", lambda *args, **kwargs: StubSource(rows))

    fever.register(register)

    record = stream("fever_v2").first(1)[0]
    assert record.answer.text == "SUPPORTS"
    assert "Sky" in str(record.metadata["evidence"][0])


def test_mathvista_filters_and_normalization(monkeypatch, tmp_path):
    rows = [
        {
            "question": "What shape is shown?",
            "options": ["Circle", "Square"],
            "answer": "A",
            "image": "diagram.png",
            "width": 512,
            "height": 512,
        },
        {
            "question": "Should be filtered",
            "options": ["A", "B"],
            "answer": "A",
            "policy_flag": True,
        },
    ]

    monkeypatch.setattr(mathvista, "HuggingFaceSource", lambda *args, **kwargs: StubSource(rows))
    mathvista.register(register)

    record = stream("mathvista").first(1)[0]
    assert record.question.text == "What shape is shown?"
    assert record.choices.to_dict()["A"] == "Circle"
    assert record.metadata["license"] == "CC-BY-SA-4.0"


def test_livecodebench_register(monkeypatch):
    rows = [
        {
            "code": "print('hello')",
            "input": "",
            "output": "hello\n",
            "language": "python",
        }
    ]
    monkeypatch.setattr(
        livecodebench, "HuggingFaceSource", lambda *args, **kwargs: StubSource(rows)
    )

    livecodebench.register(register)

    dataset_name = "livecodebench.execution"
    assert dataset_name in list_datasets()
    record = stream(dataset_name).first(1)[0]
    assert "print('hello')" in record.question.text
    assert record.answer.text == "hello\n"


def test_bbeh_parent_and_subtasks(monkeypatch):
    rows = [
        {"input": "Sort words", "target": "alpha beta", "task": "word sorting"},
        {"input": "Fix table", "target": '{"result": 42}', "task": "buggy tables"},
    ]
    monkeypatch.setattr(bbeh, "HuggingFaceSource", lambda *args, **kwargs: StubSource(rows))

    bbeh.register(register)

    assert {"bbeh", "bbeh.word_sorting", "bbeh.buggy_tables"}.issubset(set(list_datasets()))

    word_record = stream("bbeh.word_sorting").first(1)[0]
    assert word_record.metadata["task"] == "word sorting"
    assert word_record.metadata["expected_tokens"] == ("alpha", "beta")

    table_record = stream("bbeh.buggy_tables").first(1)[0]
    assert table_record.metadata["task"] == "buggy tables"
    assert table_record.metadata["result_payload"]["result"] == 42


def test_halueval_register(monkeypatch):
    rows = [
        {
            "id": "halueval-0",
            "passage": "Context A. Context B.",
            "question": "What is the answer?",
            "answer": "Answer.",
            "label": "PASS",
            "source_ds": "halueval",
            "score": 1,
        }
    ]
    monkeypatch.setattr(halueval, "HuggingFaceSource", lambda *args, **kwargs: StubSource(rows))

    halueval.register(register)

    assert "halueval" in list_datasets()
    record = stream("halueval").first(1)[0]
    assert "Context A" in record.question.text
    assert "What is the answer?" in record.question.text
    assert "Does the response contain hallucination" in record.question.text
    assert record.answer.text == "No"
    assert record.metadata["label"] == "PASS"
    assert record.metadata["original_response"] == "Answer."
    assert record.choices.to_dict() == {"A": "Yes", "B": "No"}
    assert record.metadata["license"] == "CC-BY-NC-2.0"


def test_mmlu_pro_parent_and_health(monkeypatch):
    rows = [
        {
            "question": "Which vaccine prevents disease X?",
            "options": ["Option A", "Option B"],
            "answer": "A",
            "category": "health",
        },
        {
            "question": "General question",
            "options": ["A", "B"],
            "answer": "B",
            "category": "economics",
        },
    ]
    monkeypatch.setattr(mmlu_pro, "HuggingFaceSource", lambda *args, **kwargs: StubSource(rows))

    mmlu_pro.register(register)

    assert {"mmlu_pro", "mmlu_pro.health"}.issubset(set(list_datasets()))

    record = stream("mmlu_pro.health").first(1)[0]
    assert record.metadata["category"] == "health"
    assert record.choices.to_dict()["A"] == "Option A"
    assert record.answer.text == "A"
    assert record.metadata["answer_text"] == "Option A"


def test_table_tasks_generation():
    register_table_tasks(register)

    assert {"table_arithmetic", "table_bias"}.issubset(set(list_datasets()))
    arithmetic = stream("table_arithmetic").first(1)[0]
    assert "table" in arithmetic.metadata

    bias = stream("table_bias").first(1)[0]
    assert bias.answer.text == "Mother"


def test_agieval_parent_and_subtasks(monkeypatch):
    rows_by_config = {
        config: [
            {
                "passage": None,
                "question": f"Sample prompt for {config}?",
                "choices": ["(A)No", "(B)Yes"],
                "answer": "B",
                "descriptionAnswer": "Yes is correct.",
                "other": {"exam": config},
            }
        ]
        for config in agieval._CONFIG_NAMES
    }

    def fake_hf_source(repo_id, split, config, trust_remote_code):
        return StubSource(rows_by_config[config])

    monkeypatch.setattr(agieval, "HuggingFaceSource", fake_hf_source)

    agieval.register(register)

    assert "agieval" in list_datasets()
    parent_record = stream("agieval").first(1)[0]
    assert parent_record.answer.text == "Yes"
    assert parent_record.metadata["exam"] == agieval._CONFIG_NAMES[0]

    subset_name = f"agieval.{agieval._CONFIG_NAMES[0]}"
    assert subset_name in list_datasets()
    subset_record = stream(subset_name).first(1)[0]
    assert subset_record.metadata["exam"] == agieval._CONFIG_NAMES[0]


def test_mbpp_register(monkeypatch):
    rows = [
        {
            "task_id": 42,
            "text": "Write a function that returns the cube of a number.",
            "code": "def cube(x):\n    return x ** 3",
            "test_list": ["assert cube(3) == 27"],
            "challenge_test_list": ["assert cube(-2) == -8"],
            "test_setup_code": "",
        }
    ]

    monkeypatch.setattr(mbpp, "HuggingFaceSource", lambda *args, **kwargs: StubSource(rows))

    mbpp.register(register)

    assert "mbpp" in list_datasets()
    record = stream("mbpp").first(1)[0]
    assert "cube" in record.answer.text
    assert record.metadata["task_id"] == 42
    assert "challenge_tests" in record.metadata
