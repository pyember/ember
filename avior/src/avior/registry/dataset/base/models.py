from pydantic import BaseModel
from typing import Any, Dict
from enum import Enum


class TaskType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    BINARY_CLASSIFICATION = "binary_classification"
    SHORT_ANSWER = "short_answer"
    CODE_COMPLETION = "code_completion"


class DatasetInfo(BaseModel):
    name: str
    description: str
    source: str
    task_type: TaskType


class DatasetEntry(BaseModel):
    query: str
    choices: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}
