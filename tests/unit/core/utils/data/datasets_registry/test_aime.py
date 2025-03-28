"""Unit tests for the AIME dataset implementation.

Comprehensive test suite for the AIMEPrepper class, covering:
1. Configuration handling
2. Dataset entry creation
3. Filtering functionality
4. Error handling
5. Edge cases
6. Data normalization
"""

import pytest
from typing import Dict, List, Any

from ember.core.utils.data.base.models import DatasetEntry, TaskType
from ember.core.utils.data.datasets_registry.aime import AIMEConfig, AIMEPrepper


class TestAIMEPrepper:
    """Test suite for the AIME dataset prepper."""

    def test_initialization(self):
        """Test initialization with different config types."""
        # Test with no config
        prepper1 = AIMEPrepper()
        assert prepper1.year == 2024
        assert prepper1.contest is None

        # Test with year as string
        prepper2 = AIMEPrepper("2023")
        assert prepper2.year == 2023
        assert prepper2.contest is None

        # Test with config object
        config = AIMEConfig(year=2022, contest="I")
        prepper3 = AIMEPrepper(config)
        assert prepper3.year == 2022
        assert prepper3.contest == "I"

    def test_required_keys(self):
        """Test the required keys are correctly defined."""
        prepper = AIMEPrepper()
        required_keys = prepper.get_required_keys()

        assert "ID" in required_keys
        assert "Problem" in required_keys
        assert "Answer" in required_keys
        assert len(required_keys) == 3

    def test_create_dataset_entries(self):
        """Test creating dataset entries from raw items."""
        prepper = AIMEPrepper()

        # Create a sample AIME problem
        item = {
            "ID": "2024-I-5",
            "Problem": "Find the sum of all positive integers n such that n^2 + 3n + 5 is divisible by 7.",
            "Answer": "42",
            "Solution": "Sample solution text",
        }

        entries = prepper.create_dataset_entries(item=item)

        assert len(entries) == 1
        entry = entries[0]

        assert entry.query == item["Problem"]
        assert entry.choices == {}  # No choices for short answer
        assert entry.metadata["correct_answer"] == "42"
        assert entry.metadata["problem_id"] == "2024-I-5"
        assert entry.metadata["solution"] == "Sample solution text"
        assert entry.metadata["year"] == 2024
        assert entry.metadata["contest"] == "I"
        assert entry.metadata["domain"] == "mathematics"
        assert entry.metadata["task_type"] == "short_answer"

    def test_year_filter(self):
        """Test filtering problems by year."""
        # Create a prepper that only accepts 2023 problems
        prepper = AIMEPrepper(AIMEConfig(year=2023))

        # 2024 problem should be filtered out
        item1 = {"ID": "2024-I-1", "Problem": "Problem text", "Answer": "42"}
        entries1 = prepper.create_dataset_entries(item=item1)
        assert len(entries1) == 0

        # 2023 problem should be included
        item2 = {"ID": "2023-I-1", "Problem": "Problem text", "Answer": "42"}
        entries2 = prepper.create_dataset_entries(item=item2)
        assert len(entries2) == 1

    def test_contest_filter(self):
        """Test filtering problems by contest (I or II)."""
        # Create a prepper that only accepts contest II
        prepper = AIMEPrepper(AIMEConfig(contest="II"))

        # Contest I problem should be filtered out
        item1 = {"ID": "2024-I-1", "Problem": "Problem text", "Answer": "42"}
        entries1 = prepper.create_dataset_entries(item=item1)
        assert len(entries1) == 0

        # Contest II problem should be included
        item2 = {"ID": "2024-II-1", "Problem": "Problem text", "Answer": "42"}
        entries2 = prepper.create_dataset_entries(item=item2)
        assert len(entries2) == 1

    def test_invalid_id_format(self):
        """Test handling problems with invalid ID format."""
        prepper = AIMEPrepper()

        # Item with invalid ID format should still be processed
        item = {"ID": "invalid-format", "Problem": "Problem text", "Answer": "42"}
        entries = prepper.create_dataset_entries(item=item)
        assert len(entries) == 1
        assert entries[0].metadata["year"] is None
        assert entries[0].metadata["contest"] is None

    def test_answer_normalization(self):
        """Test that answers are normalized correctly."""
        prepper = AIMEPrepper()

        # Test with numeric answer
        item1 = {"ID": "2024-I-1", "Problem": "Problem", "Answer": "042"}
        entries1 = prepper.create_dataset_entries(item=item1)
        assert entries1[0].metadata["correct_answer"] == "042"

        # Test with non-numeric answer
        item2 = {"ID": "2024-I-2", "Problem": "Problem", "Answer": "invalid"}
        entries2 = prepper.create_dataset_entries(item=item2)
        assert entries2[0].metadata["correct_answer"] == "invalid"

    def test_problem_formatting(self):
        """Test that problem text is formatted correctly."""
        prepper = AIMEPrepper()

        # Test with extra whitespace
        item = {
            "ID": "2024-I-1",
            "Problem": "  Problem with extra spaces  ",
            "Answer": "42",
        }
        entries = prepper.create_dataset_entries(item=item)
        assert entries[0].query == "Problem with extra spaces"

    def test_missing_required_keys(self):
        """Test behavior when required keys are missing."""
        prepper = AIMEPrepper()

        # Test with missing ID
        item1 = {"Problem": "Test problem", "Answer": "42"}
        with pytest.raises(KeyError):
            # Should fail when caller tries to access the missing ID key
            prepper.create_dataset_entries(item=item1)

        # Test with missing Problem
        item2 = {"ID": "2024-I-1", "Answer": "42"}
        with pytest.raises(KeyError):
            prepper.create_dataset_entries(item=item2)

        # Test with missing Answer
        item3 = {"ID": "2024-I-1", "Problem": "Test problem"}
        with pytest.raises(KeyError):
            prepper.create_dataset_entries(item=item3)

    def test_answer_out_of_range(self):
        """Test handling of answers outside the valid AIME range (0-999)."""
        prepper = AIMEPrepper()

        # Test with negative answer
        item1 = {"ID": "2024-I-1", "Problem": "Test problem", "Answer": "-5"}
        entries1 = prepper.create_dataset_entries(item=item1)
        assert entries1[0].metadata["correct_answer"] == "-5"

        # Test with answer > 999
        item2 = {"ID": "2024-I-1", "Problem": "Test problem", "Answer": "1000"}
        entries2 = prepper.create_dataset_entries(item=item2)
        assert entries2[0].metadata["correct_answer"] == "1000"

        # Test with very large number
        item3 = {"ID": "2024-I-1", "Problem": "Test problem", "Answer": "1234567890"}
        entries3 = prepper.create_dataset_entries(item=item3)
        assert entries3[0].metadata["correct_answer"] == "1234567890"

    def test_combined_filters(self):
        """Test when multiple filters are applied simultaneously."""
        # Create a prepper that only accepts 2023 contest II problems
        prepper = AIMEPrepper(AIMEConfig(year=2023, contest="II"))

        # Wrong year, wrong contest
        item1 = {"ID": "2024-I-1", "Problem": "Problem text", "Answer": "42"}
        entries1 = prepper.create_dataset_entries(item=item1)
        assert len(entries1) == 0

        # Correct year, wrong contest
        item2 = {"ID": "2023-I-1", "Problem": "Problem text", "Answer": "42"}
        entries2 = prepper.create_dataset_entries(item=item2)
        assert len(entries2) == 0

        # Wrong year, correct contest
        item3 = {"ID": "2024-II-1", "Problem": "Problem text", "Answer": "42"}
        entries3 = prepper.create_dataset_entries(item=item3)
        assert len(entries3) == 0

        # Correct year, correct contest
        item4 = {"ID": "2023-II-1", "Problem": "Problem text", "Answer": "42"}
        entries4 = prepper.create_dataset_entries(item=item4)
        assert len(entries4) == 1

    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        prepper = AIMEPrepper()

        # Test with non-string ID
        item1 = {"ID": 12345, "Problem": "Test problem", "Answer": "42"}
        entries1 = prepper.create_dataset_entries(item=item1)
        assert entries1[0].metadata["problem_id"] == "12345"

        # Test with non-string Problem
        item2 = {
            "ID": "2024-I-1",
            "Problem": ["This", "is", "not", "a", "string"],
            "Answer": "42",
        }
        # This should convert the problem to a string representation
        entries2 = prepper.create_dataset_entries(item=item2)
        assert "['This'" in entries2[0].query

        # Test with Solution as non-string
        item3 = {"ID": "2024-I-1", "Problem": "Test", "Answer": "42", "Solution": 123}
        entries3 = prepper.create_dataset_entries(item=item3)
        assert entries3[0].metadata["solution"] == 123

    def test_metadata_completeness(self):
        """Test that all expected metadata fields are present."""
        prepper = AIMEPrepper()
        item = {"ID": "2024-I-1", "Problem": "Test problem", "Answer": "42"}

        entries = prepper.create_dataset_entries(item=item)
        metadata = entries[0].metadata

        # Check all required metadata fields
        assert "correct_answer" in metadata
        assert "problem_id" in metadata
        assert "year" in metadata
        assert "contest" in metadata
        assert "domain" in metadata
        assert "task_type" in metadata
        assert "difficulty" in metadata

        # Verify values
        assert metadata["domain"] == "mathematics"
        assert metadata["task_type"] == "short_answer"
        assert metadata["difficulty"] == "challenging"
