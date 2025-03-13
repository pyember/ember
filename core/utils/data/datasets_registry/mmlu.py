from typing import Any, Dict, List, Optional

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.base.preppers import IDatasetPrepper


class MMLUConfig(BaseDatasetConfig):
    """Configuration for the MMLU dataset.

    Attributes:
        config_name (Optional[str]): Name identifier for a specific MMLU sub-dataset.
        split (Optional[str]): Data split to be used (e.g., "train", "test", etc.).
    """

    config_name: Optional[str] = None
    split: Optional[str] = None


class MMLUPrepper(IDatasetPrepper):
    """Dataset prepper for MMLU items.

    Transforms raw MMLU data into a standardized DatasetEntry.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initializes the MMLUPrepper with the provided configuration.

        Args:
            config (Optional[Any]): Configuration for MMLU.
                Can be a string (treated as config_name), MMLUConfig instance, or None.
                If None, a default MMLUConfig is created.
        """
        # Handle string configs by converting to MMLUConfig
        if isinstance(config, str):
            config = MMLUConfig(config_name=config)
        elif config is None:
            config = MMLUConfig()
        super().__init__(config)
        self.config_name: Optional[str] = self._config.config_name
        self.split: Optional[str] = self._config.split

    def get_required_keys(self) -> List[str]:
        """Returns the list of required keys for an MMLU dataset item.

        Note:
            The "subject" key is intentionally omitted to handle missing data gracefully.

        Returns:
            List[str]: The keys: "question", "choices", and "answer".
        """
        return ["question", "choices", "answer"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Creates a list containing a single DatasetEntry from a raw MMLU dataset item.

        This method converts the raw input into a standardized format with strong type
        casting and named argument invocation for clarity.

        Args:
            item (Dict[str, Any]): The raw dataset item containing keys:
                - "question": The main query as a string.
                - "choices": A list of choice strings.
                - "answer": The correct answer represented as an integer index or string.
                - "subject" (optional): The subject area of the question.

        Returns:
            List[DatasetEntry]: A list with one DatasetEntry constructed from the input.
        """
        # Convert the question to a string explicitly.
        question: str = str(item["question"])

        # Process 'choices': Map each choice to an uppercase letter starting from 'A'.
        choices_list: List[Any] = item["choices"]
        choices: Dict[str, str] = {
            chr(65 + i): str(choice) for i, choice in enumerate(choices_list)
        }

        # Determine the correct answer, converting an integer index to the corresponding letter if needed.
        raw_answer: Any = item["answer"]
        if isinstance(raw_answer, int):
            correct_answer: str = chr(65 + raw_answer)
        else:
            correct_answer = str(raw_answer)

        # Handle 'subject' gracefully: convert to string if present.
        subject_raw: Optional[Any] = item.get("subject")
        subject: Optional[str] = str(subject_raw) if subject_raw is not None else None

        # Construct the DatasetEntry using named parameters.
        entry: DatasetEntry = DatasetEntry(
            query=question,
            choices=choices,
            metadata={
                "correct_answer": correct_answer,
                "subject": subject,
                "config_name": self.config_name,
            },
        )
        return [entry]
