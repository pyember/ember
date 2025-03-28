"""GPQA Diamond dataset for PhD-level science questions.

This module implements the prepper and configuration for the GPQA Diamond dataset,
which contains graduate-level, 'Google-proof' multiple-choice science questions in
physics, chemistry, and biology.
"""

from typing import Any, Dict, List, Optional

from ember.core.utils.data.base.config import BaseDatasetConfig
from ember.core.utils.data.base.models import DatasetEntry, TaskType
from ember.core.utils.data.base.preppers import IDatasetPrepper


class GPQAConfig(BaseDatasetConfig):
    """Configuration for the GPQA dataset.

    Controls loading options for the GPQA PhD-level science questions.
    """

    subset: str = "gpqa_diamond"  # Default to Diamond subset
    difficulty: Optional[str] = None  # Optional filter by difficulty level
    domain: Optional[str] = None  # Optional filter by scientific domain


class GPQAPrepper(IDatasetPrepper):
    """Prepper for GPQA Diamond science questions.

    Transforms HuggingFace GPQA dataset entries into DatasetEntry format,
    handling the multiple-choice structure with correct and incorrect answers.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the GPQA prepper with optional configuration.

        Args:
            config: Either a string (subset name), GPQAConfig instance, or None.
                   If None, defaults to Diamond subset.
        """
        if isinstance(config, str):
            config = GPQAConfig(subset=config)
        elif config is None:
            config = GPQAConfig()
        super().__init__(config)

        self.subset = self._config.subset
        self.difficulty = self._config.difficulty
        self.domain = self._config.domain

    def get_required_keys(self) -> List[str]:
        """Return required keys for GPQA dataset items.

        Returns:
            List of required fields for processing.
        """
        return ["question_id", "question", "choices", "answer", "domain", "difficulty"]

    def create_dataset_entries(self, *, item: Dict[str, Any]) -> List[DatasetEntry]:
        """Create a DatasetEntry from a GPQA question.

        Transforms a raw GPQA dataset item into Ember's DatasetEntry format,
        applying filters based on domain and difficulty if specified.

        Args:
            item: Raw dataset item containing:
                - "question_id": Unique identifier
                - "question": The question text
                - "choices": Dict of answer choices keyed by letter (A, B, C, D)
                - "answer": Correct answer letter
                - "domain": Scientific domain (physics, chemistry, biology)
                - "difficulty": Difficulty rating

        Returns:
            List with one DatasetEntry if it passes filters, empty list otherwise
        """
        # Extract and validate required fields
        question_id = str(item["question_id"])
        question_text = str(item["question"])
        choices_dict = item["choices"]
        correct_answer = str(item["answer"])
        domain = str(item.get("domain", ""))
        difficulty = str(item.get("difficulty", ""))

        # Apply domain filter if specified
        if self.domain and domain.lower() != self.domain.lower():
            return []  # Skip if domain doesn't match

        # Apply difficulty filter if specified
        if self.difficulty and difficulty.lower() != self.difficulty.lower():
            return []  # Skip if difficulty doesn't match

        # Format question choices consistently
        formatted_choices = {}
        for letter, choice_text in choices_dict.items():
            formatted_choices[letter] = str(choice_text).strip()

        # Format question text
        formatted_question = question_text.strip()

        # Create dataset entry
        return [
            DatasetEntry(
                query=formatted_question,
                choices=formatted_choices,  # Multiple choice format
                metadata={
                    "correct_answer": correct_answer,
                    "question_id": question_id,
                    "domain": domain,
                    "difficulty": difficulty,
                    "task_type": "multiple_choice",
                    "dataset": "gpqa",
                    "subset": self.subset,
                },
            )
        ]


# The GPQA dataset will be registered in the initialize_registry function
# in the ember.core.utils.data.registry module
