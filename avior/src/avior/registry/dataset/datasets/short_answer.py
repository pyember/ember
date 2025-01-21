from typing import List, Dict, Any
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
from src.avior.registry.dataset.base.models import DatasetEntry


class ShortAnswerPrepper(IDatasetPrepper):
    def get_required_keys(self) -> List[str]:
        return ["question", "answer"]

    def create_dataset_entries(self, item: Dict[str, Any]) -> List[DatasetEntry]:
        return [
            DatasetEntry(
                query=item["question"],
                choices={},
                metadata={"gold_answer": item["answer"]},
            )
        ]
