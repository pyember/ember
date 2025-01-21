from typing import List, Dict, Any
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
from src.avior.registry.dataset.base.models import DatasetEntry


class CommonsenseQAPrepper(IDatasetPrepper):
    def get_required_keys(self) -> List[str]:
        return ["question", "choices", "answerKey"]

    def create_dataset_entries(self, item: Dict[str, Any]) -> List[DatasetEntry]:
        question = item["question"]
        choices = {
            c["label"]: c["text"] for c in item["choices"] if isinstance(c, dict)
        }
        correct_answer = item["answerKey"]
        return [
            DatasetEntry(
                query=question,
                choices=choices,
                metadata={"correct_answer": correct_answer},
            )
        ]
