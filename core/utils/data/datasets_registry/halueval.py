from typing import Any, Dict, List, Optional

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.base.preppers import IDatasetPrepper


class HaluEvalConfig(BaseDatasetConfig):
    """Configuration for HaluEval dataset with specific fields.

    Attributes:
        config_name (Optional[str]): Sub-dataset name (default "qa").
        split (Optional[str]): Data split to use (default "data").
    """

    config_name: Optional[str] = "qa"
    split: Optional[str] = "data"


class HaluEvalPrepper(IDatasetPrepper):
    """Dataset prepper for HaluEval items.

    Transforms a raw HaluEval dataset item into structured DatasetEntry objects.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initializes the HaluEvalPrepper with the provided configuration.

        Args:
            config (Optional[Any]): Configuration for HaluEval.
                Can be a string (treated as config_name), HaluEvalConfig instance, or None.
                If None, a default HaluEvalConfig is created.
        """
        # Handle string configs by converting to HaluEvalConfig
        if isinstance(config, str):
            config = HaluEvalConfig(config_name=config)
        elif config is None:
            config = HaluEvalConfig()
        super().__init__(config)
        self.config_name: Optional[str] = self._config.config_name
        self.split: Optional[str] = self._config.split

    def get_required_keys(self) -> List[str]:
        """Retrieves the required keys for a HaluEval dataset item.

        Returns:
            List[str]: The list of required keys: "knowledge", "question",
            "right_answer", and "hallucinated_answer".
        """
        return ["knowledge", "question", "right_answer", "hallucinated_answer"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Creates dataset entries from a raw HaluEval item.

        Converts the candidate answers into two separate entries for evaluation:
        one for the correct (non-hallucinated) answer and one for the hallucinated answer.

        Args:
            item (Dict[str, Any]): A dictionary with HaluEval data containing the keys:
                "knowledge", "question", "right_answer", "hallucinated_answer".

        Returns:
            List[DatasetEntry]: A list containing two DatasetEntry instances.
        """
        knowledge: str = str(item["knowledge"])
        question: str = str(item["question"])
        right_answer: str = str(item["right_answer"])
        hallucinated_answer: str = str(item["hallucinated_answer"])

        not_hallucinated_entry: DatasetEntry = self._build_dataset_entry(
            knowledge=knowledge,
            question=question,
            candidate_answer=right_answer,
            correct_choice="A",
        )

        hallucinated_entry: DatasetEntry = self._build_dataset_entry(
            knowledge=knowledge,
            question=question,
            candidate_answer=hallucinated_answer,
            correct_choice="B",
        )

        return [not_hallucinated_entry, hallucinated_entry]

    def _build_dataset_entry(
        self,
        *,
        knowledge: str,
        question: str,
        candidate_answer: str,
        correct_choice: str,
    ) -> DatasetEntry:
        """Builds a DatasetEntry with a formatted query and corresponding metadata.

        Args:
            knowledge (str): The supporting knowledge.
            question (str): The question being asked.
            candidate_answer (str): The candidate answer to evaluate.
            correct_choice (str): The correct label ("A" for non-hallucinated, "B" for hallucinated).

        Returns:
            DatasetEntry: The constructed dataset entry.
        """
        query_text: str = (
            f"Knowledge: {knowledge}\n"
            f"Question: {question}\n"
            f"Candidate Answer: {candidate_answer}. "
            "Is this candidate answer supported by the provided knowledge?"
        )

        return DatasetEntry(
            query=query_text,
            choices={"A": "Not Hallucinated", "B": "Hallucinated"},
            metadata={"correct_answer": correct_choice},
        )
