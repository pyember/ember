"""Codeforces competitive programming dataset.

This module implements the prepper and configuration for the Codeforces dataset,
which contains competitive programming problems for code generation and evaluation.
"""

from typing import Any, Dict, List, Optional, Tuple

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper


class CodeForcesConfig(BaseDatasetConfig):
    """Configuration for the Codeforces dataset.

    Controls filtering and loading options for competitive programming problems.
    """

    difficulty_range: Optional[Tuple[int, int]] = None  # Min/max difficulty
    tags: Optional[List[str]] = None  # Problem tags for filtering
    limit: Optional[int] = None  # Max number of problems


class CodeForcesPrepper(IDatasetPrepper):
    """Prepper for Codeforces programming problems.

    Transforms HuggingFace Codeforces dataset entries into DatasetEntry format,
    handling problem statements, input specifications, and test cases.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the Codeforces prepper with optional configuration.

        Args:
            config: CodeForcesConfig instance or None for defaults.
        """
        if config is None:
            config = CodeForcesConfig()
        super().__init__(config)

        # Store configuration for filtering
        self.difficulty_range = self._config.difficulty_range
        self.tags = self._config.tags
        self.limit = self._config.limit
        self.processed_count = 0

    def get_required_keys(self) -> List[str]:
        """Return required keys for Codeforces dataset items.

        Returns:
            List of required fields for the Codeforces dataset format.
        """
        # Note: The open-r1/codeforces dataset uses a different schema than our test fixtures
        # In the actual dataset, fields use these keys
        return [
            "id",
            "title",
            "description",
            "input_format",
            "output_format",
            "examples",
        ]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create a DatasetEntry from a Codeforces problem.

        Transforms a raw Codeforces problem from the open-r1/codeforces dataset format
        into Ember's standardized DatasetEntry format, applying filtering if specified.

        Args:
            item: Raw dataset item containing problem details from open-r1/codeforces:
                - "id": Problem identifier (e.g., "852/A")
                - "title": Problem title
                - "description": Problem description
                - "input_format": Input format description
                - "output_format": Output format requirements
                - "examples": List of test cases with inputs and outputs
                - Additional fields available but not required

        Returns:
            DatasetEntry with the problem as query and test cases in metadata,
            or empty list if the problem doesn't pass filters
        """
        # Check limit first
        if self.limit is not None and self.processed_count >= self.limit:
            return []

        # Extract problem data
        problem_id = str(item["id"])
        name = str(item["title"])
        statement = str(item["description"])
        input_spec = str(item.get("input_format", ""))
        output_spec = str(item.get("output_format", ""))

        # Get difficulty (using provided or default to 0)
        difficulty = item.get("difficulty", 0)

        # Get tags (using provided or default to empty list)
        tags = item.get("tags", [])

        # Get test cases - named "examples" in the dataset format
        test_cases = item.get("examples", [])

        # Apply difficulty range filter if specified
        if self.difficulty_range:
            min_diff, max_diff = self.difficulty_range
            if difficulty < min_diff or difficulty > max_diff:
                return []  # Skip if outside difficulty range

        # Apply tags filter if specified
        if self.tags:
            # Check if any of the required tags are present
            if not any(tag in tags for tag in self.tags):
                return []  # Skip if no matching tags

        # Format problem text in a standard way
        formatted_problem = (
            f"# {name}\n\n"
            f"{statement}\n\n"
            f"## Input Specification\n{input_spec}\n\n"
            f"## Output Specification\n{output_spec}"
        ).strip()

        # Increment the processed count
        self.processed_count += 1

        # Create and return dataset entry
        return [
            DatasetEntry(
                query=formatted_problem,
                choices={},  # No choices for code generation problems
                metadata={
                    "problem_id": problem_id,
                    "name": name,
                    "difficulty": difficulty,
                    "tags": tags,
                    "test_cases": test_cases,
                    "task_type": "code_completion",
                    "dataset": "codeforces",
                    "input_specification": input_spec,
                    "output_specification": output_spec,
                },
            )
        ]


# The Codeforces dataset will be registered in the initialize_registry function
# in the ember.core.utils.data.registry module
