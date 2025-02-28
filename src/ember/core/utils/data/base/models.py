from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator


class TaskType(str, Enum):
    """Enumeration of dataset task types.

    Attributes:
        MULTIPLE_CHOICE (str): Multiple-choice question type.
        BINARY_CLASSIFICATION (str): Binary classification task type.
        SHORT_ANSWER (str): Short answer task type.
        CODE_COMPLETION (str): Code completion task type.
    """

    MULTIPLE_CHOICE: str = "multiple_choice"
    BINARY_CLASSIFICATION: str = "binary_classification"
    SHORT_ANSWER: str = "short_answer"
    CODE_COMPLETION: str = "code_completion"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return self.value


class DatasetInfo(BaseModel):
    """Model representing essential dataset information.

    Attributes:
        name (str): Name of the dataset.
        description (str): Brief description of the dataset.
        source (str): Origin or provider of the dataset.
        task_type (TaskType): The type of task associated with this dataset.
    """

    name: str
    description: str
    source: str
    task_type: TaskType

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Validate that the name is not empty."""
        if not value or not value.strip():
            raise ValueError("Dataset name cannot be empty")
        return value


class DatasetEntry(BaseModel):
    """Model for a single dataset entry.

    This encapsulates an entry's query, potential answer choices, and related metadata.

    Attributes:
        query (str): The query prompt for the dataset entry.
        choices (Dict[str, str]): A mapping of choice identifiers to choice texts.
        metadata (Dict[str, Any]): Additional metadata associated with the entry.
    """

    query: str
    choices: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Validate that the query is not empty."""
        if not value or not value.strip():
            raise ValueError("Query cannot be empty")
        return value
