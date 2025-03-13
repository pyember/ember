from typing import List, Dict, Any

from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.base.models import DatasetEntry


class ShortAnswerPrepper(IDatasetPrepper):
    """Prepper for short answer dataset entries.

    This class converts a raw dataset item containing a question and its
    corresponding answer into a standardized DatasetEntry suitable for further
    processing.
    """

    def get_required_keys(self) -> List[str]:
        """Returns the list of keys required for a short answer dataset item.

        Returns:
            List[str]: A list containing the required keys:
                - "question": The text of the question.
                - "answer": The correct answer.
        """
        return ["question", "answer"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Creates a list with a single standardized DatasetEntry from the provided item.

        This method extracts the question and the answer from the dataset item,
        ensuring both are represented as strings, and constructs a DatasetEntry
        with empty choices and metadata that includes the gold answer.

        Args:
            item (Dict[str, Any]): A dictionary representing the dataset item, which
                must include the keys "question" and "answer".

        Returns:
            List[DatasetEntry]: A list containing one DatasetEntry encapsulating the
            processed question and its corresponding gold answer.
        """
        question: str = str(item["question"])
        gold_answer: str = str(item["answer"])

        dataset_entry: DatasetEntry = DatasetEntry(
            query=question,
            choices={},  # Short answer entries do not include multiple choices.
            metadata={"gold_answer": gold_answer},
        )
        return [dataset_entry]
