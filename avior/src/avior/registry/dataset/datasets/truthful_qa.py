from typing import List, Dict, Any
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
from src.avior.registry.dataset.base.models import DatasetEntry


class TruthfulQAPrepper(IDatasetPrepper):
    def get_required_keys(self) -> List[str]:
        return ["question", "mc1_targets"]

    def create_dataset_entries(self, item: Dict[str, Any]) -> List[DatasetEntry]:
        question = item["question"]
        mc1_targets = item["mc1_targets"]

        choices = {
            chr(65 + i): choice for i, choice in enumerate(mc1_targets["choices"])
        }
        correct_answer = next(
            (
                chr(65 + i)
                for i, label in enumerate(mc1_targets["labels"])
                if label == 1
            ),
            None,
        )
        return [
            DatasetEntry(
                query=question,
                choices=choices,
                metadata={"correct_answer": correct_answer},
            )
        ]
