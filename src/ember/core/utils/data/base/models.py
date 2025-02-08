from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field


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
