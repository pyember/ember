from typing import List, Dict, Any
from ember.core.utils.data.base.preppers import IDatasetPrepper
from ember.core.utils.data.base.models import DatasetEntry


class CommonsenseQAPrepper(IDatasetPrepper):
    """Dataset prepper for processing Commonsense QA data entries.

    This class transforms a raw Commonsense QA dataset item into a standardized format.
    """

    def get_required_keys(self) -> List[str]:
        """Return a list of keys required in a Commonsense QA dataset item.

        Returns:
            List[str]: The required keys: 'question', 'choices', and 'answerKey'.
        """
        return ["question", "choices", "answerKey"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create dataset entries from a raw Commonsense QA dictionary item.

        Extracts the question, choices, and correct answer from the input dictionary
        and returns a list containing a single DatasetEntry initialized with these values.

        Args:
            item (Dict[str, Any]): A dictionary representing a Commonsense QA dataset item.
                It must include the keys 'question', 'choices', and 'answerKey'.

        Returns:
            List[DatasetEntry]: A list with one DatasetEntry constructed from the provided item.
        """
        question: str = str(item["question"])
        choices: Dict[str, str] = self._parse_choices(choices_data=item["choices"])
        correct_answer: str = str(item["answerKey"])

        return [
            DatasetEntry(
                query=question,
                choices=choices,
                metadata={"correct_answer": correct_answer},
            )
        ]

    def _parse_choices(self, *, choices_data: Any) -> Dict[str, str]:
        """Parse the raw choices into a dictionary mapping labels to texts.

        Each choice is expected to be a dictionary containing the keys 'label' and 'text'.
        If choices_data is not a list, a ValueError is raised.

        Args:
            choices_data (Any): The raw choices data extracted from the dataset item.

        Raises:
            ValueError: If 'choices_data' is not a list.

        Returns:
            Dict[str, str]: A dictionary mapping each choice's label to its corresponding text.
        """
        if not isinstance(choices_data, list):
            raise ValueError(
                "Expected 'choices' to be a list, got {} instead.".format(
                    type(choices_data).__name__
                )
            )

        parsed_choices: Dict[str, str] = {}
        for choice in choices_data:
            if isinstance(choice, dict) and "label" in choice and "text" in choice:
                label: str = str(choice["label"])
                text: str = str(choice["text"])
                parsed_choices[label] = text
        return parsed_choices
