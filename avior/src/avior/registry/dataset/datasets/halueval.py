from typing import List, Dict, Any
from src.avior.registry.dataset.base.preppers import IDatasetPrepper
from src.avior.registry.dataset.base.models import DatasetEntry
from typing import Optional
from src.avior.registry.dataset.base.config import BaseDatasetConfig


class HaluEvalConfig(BaseDatasetConfig):
    """
    Represents HaluEval-specific config fields, e.g. a sub-dataset name on HF.
    """
    config_name: Optional[str] = "qa"
    split: Optional[str] = "data"


class HaluEvalPrepper(IDatasetPrepper):
    def get_required_keys(self) -> List[str]:
        return ["knowledge", "question", "right_answer", "hallucinated_answer"]

    def create_dataset_entries(self, item: Dict[str, Any]) -> List[DatasetEntry]:
        knowledge = item["knowledge"]
        question = item["question"]
        right_answer = item["right_answer"]
        hallucinated_answer = item["hallucinated_answer"]

        not_hallucinated = DatasetEntry(
            query=(
                f"Knowledge: {knowledge}\nQuestion: {question}\n"
                f"Candidate Answer: {right_answer}. "
                "Is this candidate answer supported by the provided knowledge?"
            ),
            choices={"A": "Not Hallucinated", "B": "Hallucinated"},
            metadata={"correct_answer": "A"},
        )

        hallucinated = DatasetEntry(
            query=(
                f"Knowledge: {knowledge}\nQuestion: {question}\n"
                f"Candidate Answer: {hallucinated_answer}. "
                "Is this candidate answer supported by the provided knowledge?"
            ),
            choices={"A": "Not Hallucinated", "B": "Hallucinated"},
            metadata={"correct_answer": "B"},
        )
        return [not_hallucinated, hallucinated]
