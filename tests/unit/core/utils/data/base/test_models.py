"""Unit tests for data model classes."""

import unittest
from typing import Any, Dict, Optional

from src.ember.core.utils.data.base.models import TaskType, DatasetInfo, DatasetEntry


class TestTaskType(unittest.TestCase):
    """Test cases for the TaskType enumeration."""

    def test_enum_values(self) -> None:
        """TaskType should contain the expected enumeration values."""
        # Assert that all expected values are present
        self.assertEqual("multiple_choice", TaskType.MULTIPLE_CHOICE)
        self.assertEqual("binary_classification", TaskType.BINARY_CLASSIFICATION)
        self.assertEqual("short_answer", TaskType.SHORT_ANSWER)
        self.assertEqual("code_completion", TaskType.CODE_COMPLETION)

    def test_enum_equality(self) -> None:
        """TaskType instances should support equality comparison."""
        # Arrange
        task1 = TaskType.MULTIPLE_CHOICE
        task2 = TaskType.MULTIPLE_CHOICE
        task3 = TaskType.SHORT_ANSWER

        # Assert
        self.assertEqual(task1, task2)
        self.assertNotEqual(task1, task3)

    def test_string_representation(self) -> None:
        """TaskType should convert to expected string values."""
        # Arrange
        task = TaskType.MULTIPLE_CHOICE

        # Act & Assert
        self.assertEqual("multiple_choice", str(task))
        self.assertEqual("multiple_choice", task.value)


class TestDatasetInfo(unittest.TestCase):
    """Test cases for the DatasetInfo model."""

    def test_init_required_fields(self) -> None:
        """DatasetInfo should initialize with required fields."""
        # Arrange
        name = "test_dataset"
        description = "Test dataset description"
        source = "test_source"
        task_type = TaskType.MULTIPLE_CHOICE

        # Act
        dataset_info = DatasetInfo(
            name=name, description=description, source=source, task_type=task_type
        )

        # Assert
        self.assertEqual(name, dataset_info.name)
        self.assertEqual(description, dataset_info.description)
        self.assertEqual(source, dataset_info.source)
        self.assertEqual(task_type, dataset_info.task_type)

    def test_model_validation_name(self) -> None:
        """DatasetInfo should validate the name field."""
        # Arrange & Act & Assert
        with self.assertRaises(ValueError):
            DatasetInfo(
                name="",  # Empty name should fail validation
                description="Test description",
                source="test_source",
                task_type=TaskType.MULTIPLE_CHOICE,
            )

    def test_model_serialization(self) -> None:
        """DatasetInfo should serialize to and deserialize from JSON correctly."""
        # Arrange
        original = DatasetInfo(
            name="test_dataset",
            description="Test dataset description",
            source="test_source",
            task_type=TaskType.MULTIPLE_CHOICE,
        )

        # Act
        json_str = original.model_dump_json()
        deserialized = DatasetInfo.model_validate_json(json_str)

        # Assert
        self.assertEqual(original.name, deserialized.name)
        self.assertEqual(original.description, deserialized.description)
        self.assertEqual(original.source, deserialized.source)
        self.assertEqual(original.task_type, deserialized.task_type)

    def test_task_type_serialization(self) -> None:
        """DatasetInfo should correctly serialize and deserialize TaskType enum values."""
        # Arrange
        for task_type in TaskType:
            # Act
            info = DatasetInfo(
                name="test", description="test", source="test", task_type=task_type
            )
            json_str = info.model_dump_json()
            deserialized = DatasetInfo.model_validate_json(json_str)

            # Assert
            self.assertEqual(task_type, deserialized.task_type)


class TestDatasetEntry(unittest.TestCase):
    """Test cases for the DatasetEntry model."""

    def test_init_with_minimal_fields(self) -> None:
        """DatasetEntry should initialize with just the query field."""
        # Arrange
        query = "What is the capital of France?"

        # Act
        entry = DatasetEntry(query=query)

        # Assert
        self.assertEqual(query, entry.query)
        self.assertEqual({}, entry.choices)
        self.assertEqual({}, entry.metadata)

    def test_init_with_all_fields(self) -> None:
        """DatasetEntry should initialize with all fields provided."""
        # Arrange
        query = "What is the capital of France?"
        choices = {"A": "Paris", "B": "London", "C": "Berlin"}
        metadata = {"correct_answer": "A", "difficulty": "easy"}

        # Act
        entry = DatasetEntry(query=query, choices=choices, metadata=metadata)

        # Assert
        self.assertEqual(query, entry.query)
        self.assertEqual(choices, entry.choices)
        self.assertEqual(metadata, entry.metadata)

    def test_model_validation_query(self) -> None:
        """DatasetEntry should validate that query is non-empty."""
        # Arrange & Act & Assert
        with self.assertRaises(ValueError):
            DatasetEntry(query="")  # Empty query should fail validation

    def test_model_serialization(self) -> None:
        """DatasetEntry should serialize to and deserialize from JSON correctly."""
        # Arrange
        original = DatasetEntry(
            query="What is the capital of France?",
            choices={"A": "Paris", "B": "London", "C": "Berlin"},
            metadata={"correct_answer": "A", "difficulty": "easy"},
        )

        # Act
        json_str = original.model_dump_json()
        deserialized = DatasetEntry.model_validate_json(json_str)

        # Assert
        self.assertEqual(original.query, deserialized.query)
        self.assertEqual(original.choices, deserialized.choices)
        self.assertEqual(original.metadata, deserialized.metadata)

    def test_field_defaults(self) -> None:
        """DatasetEntry should use empty dictionaries as defaults for choices and metadata."""
        # Arrange & Act
        entry = DatasetEntry(query="Test query")

        # Assert
        self.assertEqual({}, entry.choices)
        self.assertEqual({}, entry.metadata)


if __name__ == "__main__":
    unittest.main()
