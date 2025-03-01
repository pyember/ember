from typing import List, Dict, Any, Optional
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.base.models import DatasetEntry


class TruthfulQAPrepper(IDatasetPrepper):
    """Dataset prepper for the TruthfulQA dataset.

    This class transforms a raw TruthfulQA dataset item into a standardized DatasetEntry.
    """

    def get_required_keys(self) -> List[str]:
        """Retrieve the required keys from a dataset item.

        Returns:
            List[str]: A list containing "question" and "mc1_targets".
        """
        return ["question", "mc1_targets"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create a list of DatasetEntry objects from a raw TruthfulQA dataset item.

        This method extracts the question and multiple-choice targets from the input item,
        transforms the choices to letter mappings (starting from 'A'), and identifies the correct
        answer based on label flags. The processed information is packaged into a DatasetEntry.

        Args:
            item (Dict[str, Any]): The raw dataset item with expected keys:
                - "question": A string representing the query.
                - "mc1_targets": A dictionary containing:
                    - "choices": A list of choice strings.
                    - "labels": A list of integers where a value of 1 indicates the correct answer.

        Returns:
            List[DatasetEntry]: A list containing a single, processed DatasetEntry.
        """
        question: str = str(item["question"])
        mc1_targets: Dict[str, Any] = item["mc1_targets"]

        choices: Dict[str, str] = {
            chr(65 + index): choice
            for index, choice in enumerate(mc1_targets["choices"])
        }
        correct_answer: Optional[str] = next(
            (
                chr(65 + index)
                for index, label in enumerate(mc1_targets["labels"])
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
