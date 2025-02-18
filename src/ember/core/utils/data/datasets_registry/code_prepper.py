from typing import List, Dict, Any

from src.ember.core.utils.data.base.preppers import IDatasetPrepper
from src.ember.core.utils.data.base.models import DatasetEntry


class CodePrepper(IDatasetPrepper):
    """Dataset prepper for code-based datasets.

    This class converts a dictionary containing a code prompt along with its
    associated tests and programming language into a standardized DatasetEntry.

    Attributes:
        None.
    """

    def get_required_keys(self) -> List[str]:
        """Retrieves the list of keys required in the dataset item.

        Returns:
            List[str]: A list of required key names: "prompt", "tests", and "language".
        """
        return ["prompt", "tests", "language"]

    def create_dataset_entries(self, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Generates dataset entries from the provided item.

        Extracts the required fields ('prompt', 'tests', and 'language') using
        named variable assignments and constructs a DatasetEntry using explicit,
        keyword-based argument invocation.

        Args:
            item (Dict[str, Any]): A dictionary that represents the dataset item.
                It must include the keys "prompt", "tests", and "language".

        Returns:
            List[DatasetEntry]: A list containing one DatasetEntry that includes the
            query and its corresponding metadata.
        """
        # Extract the required fields with explicit type annotations.
        prompt: str = item["prompt"]
        tests: Any = item["tests"]
        language: str = item["language"]

        # Compose metadata with strong typing.
        metadata: Dict[str, Any] = {"tests": tests, "language": language}

        # Instantiate and return a DatasetEntry using named method invocation.
        return [DatasetEntry(query=prompt, choices={}, metadata=metadata)]
