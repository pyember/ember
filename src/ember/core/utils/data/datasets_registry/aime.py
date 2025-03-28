"""American Invitational Mathematics Examination (AIME) dataset.

This module implements the prepper and configuration for the AIME dataset,
which contains challenging math competition problems from the American Invitational
Mathematics Examination.
"""

from typing import Any, Dict, List, Optional

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.base.preppers import IDatasetPrepper


class AIMEConfig(BaseDatasetConfig):
    """Configuration for the AIME dataset.

    Controls filtering and loading options for the AIME math competition dataset.
    """

    year: Optional[int] = 2024  # Default to 2024
    contest: Optional[str] = None  # 'I' or 'II' for specific contest


class AIMEPrepper(IDatasetPrepper):
    """Prepper for AIME competition math problems.

    Transforms HuggingFace AIME dataset entries containing American Invitational
    Mathematics Examination problems into Ember's standardized DatasetEntry format.
    AIME problems are challenging math problems with integer answers from 0-999.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the AIME prepper with configuration.

        Args:
            config: Configuration that can be:
                - A string with year (e.g., "2024")
                - An AIMEConfig instance
                - None (defaults to all 2024 problems)
        """
        if isinstance(config, str) and config.isdigit():
            config = AIMEConfig(year=int(config))
        elif config is None:
            config = AIMEConfig()
        super().__init__(config)
        self.year = self._config.year
        self.contest = self._config.contest

    def get_required_keys(self) -> List[str]:
        """Return required keys for AIME dataset items.

        Returns:
            List containing ID, Problem, and Answer fields
        """
        return ["ID", "Problem", "Answer"]

    def _parse_problem_id(self, problem_id: str) -> Dict[str, Any]:
        """Extract components from AIME problem ID.

        Parses standard AIME ID format "YYYY-C-N" where:
        - YYYY = year (e.g., 2024)
        - C = contest number (I or II)
        - N = problem number within contest (1-15)

        Args:
            problem_id: String identifier for the problem

        Returns:
            Dictionary with extracted year and contest, or None if not parseable
        """
        result = {"year": None, "contest": None, "number": None}
        parts = problem_id.split("-")

        if len(parts) >= 3:
            if parts[0].isdigit():
                result["year"] = int(parts[0])
            result["contest"] = parts[1]
            if parts[2].isdigit():
                result["number"] = int(parts[2])

        return result

    def _should_include_problem(self, parsed_id: Dict[str, Any]) -> bool:
        """Determine if a problem should be included based on filters.

        Args:
            parsed_id: Dictionary with problem metadata including year and contest

        Returns:
            True if problem should be included, False if it should be filtered out
        """
        # Skip if year filter is set and doesn't match
        if (
            parsed_id["year"] is not None
            and self.year is not None
            and parsed_id["year"] != self.year
        ):
            return False

        # Skip if contest filter is set and doesn't match
        if (
            parsed_id["contest"] is not None
            and self.contest is not None
            and parsed_id["contest"] != self.contest
        ):
            return False

        return True

    def _normalize_answer(self, answer: str) -> str:
        """Normalize and validate AIME problem answers.

        AIME answers should be integers from 0-999, but this function
        preserves formatting and handles invalid inputs gracefully.

        Args:
            answer: The raw answer string

        Returns:
            Normalized answer string
        """
        try:
            # Convert to int but preserve original string representation
            # This ensures "042" stays as "042" rather than becoming "42"
            answer_int = int(answer)
            if not (0 <= answer_int <= 999):
                # Outside valid AIME range, but preserve as string
                return answer
            return answer
        except ValueError:
            # Not a valid integer, keep as-is for downstream handling
            return answer

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create a DatasetEntry from an AIME problem.

        Transforms a raw AIME problem entry into Ember's DatasetEntry format,
        applying filtering based on year and contest if specified.

        Input format from HuggingFace dataset:
        {
            "ID": "2024-I-1",           # Year-Contest-Problem# format
            "Problem": "Find the...",    # LaTeX-formatted text
            "Answer": "42",             # Integer (0-999)
            "Solution": "We start..."   # Optional solution text
        }

        Args:
            item: Raw dataset item containing problem data in the format above

        Returns:
            List with one DatasetEntry if it passes filters, empty list otherwise.
            Empty list is returned when:
            - Year filter is set and doesn't match item's year
            - Contest filter is set and doesn't match item's contest
        """
        # Convert all inputs to appropriate types
        problem_id = str(item["ID"])
        problem_text = str(item["Problem"])
        answer = str(item["Answer"])
        solution = item.get("Solution", "")

        # Parse problem ID into components
        parsed_id = self._parse_problem_id(problem_id)

        # Check if problem should be included based on filters
        if not self._should_include_problem(parsed_id):
            return []

        # Process the problem content
        formatted_problem = problem_text.strip()
        normalized_answer = self._normalize_answer(answer)

        # Create and return dataset entry
        return [
            DatasetEntry(
                query=formatted_problem,
                choices={},  # No choices for short answer problems
                metadata={
                    "correct_answer": normalized_answer,
                    "solution": solution,
                    "problem_id": problem_id,
                    "year": parsed_id["year"],
                    "contest": parsed_id["contest"],
                    "task_type": "short_answer",
                    "domain": "mathematics",
                    "difficulty": "challenging",
                },
            )
        ]


# The AIME dataset will be registered in the initialize_registry function
# in the ember.core.utils.data.registry module
