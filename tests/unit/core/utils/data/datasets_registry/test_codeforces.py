"""Unit tests for the Codeforces dataset prepper."""

import unittest

from ember.core.utils.data.datasets_registry.codeforces import (
    CodeForcesConfig,
    CodeForcesPrepper,
)


class TestCodeForcesConfig(unittest.TestCase):
    """Test cases for the CodeForcesConfig class."""

    def test_init_default_values(self) -> None:
        """CodeForcesConfig should initialize with default values."""
        # Arrange & Act
        config = CodeForcesConfig()

        # Assert
        self.assertIsNone(config.difficulty_range)
        self.assertIsNone(config.tags)
        self.assertIsNone(config.limit)

    def test_init_custom_values(self) -> None:
        """CodeForcesConfig should initialize with custom values."""
        # Arrange
        difficulty_range = (800, 1500)
        tags = ["implementation", "dp"]
        limit = 100

        # Act
        config = CodeForcesConfig(
            difficulty_range=difficulty_range, tags=tags, limit=limit
        )

        # Assert
        self.assertEqual(difficulty_range, config.difficulty_range)
        self.assertEqual(tags, config.tags)
        self.assertEqual(limit, config.limit)

    def test_model_serialization(self) -> None:
        """CodeForcesConfig should serialize to and deserialize from JSON correctly."""
        # Arrange
        original = CodeForcesConfig(
            difficulty_range=(800, 1500), tags=["implementation", "dp"], limit=100
        )

        # Act
        json_str = original.model_dump_json()
        deserialized = CodeForcesConfig.model_validate_json(json_str)

        # Assert
        self.assertEqual(original.difficulty_range, deserialized.difficulty_range)
        self.assertEqual(original.tags, deserialized.tags)
        self.assertEqual(original.limit, deserialized.limit)


class TestCodeForcesPrepper(unittest.TestCase):
    """Test cases for the CodeForcesPrepper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a default prepper
        self.default_prepper = CodeForcesPrepper()

        # Create a prepper with custom config
        self.config = CodeForcesConfig(
            difficulty_range=(800, 1500), tags=["implementation"], limit=10
        )
        self.configured_prepper = CodeForcesPrepper(config=self.config)

        # Create test data matching the open-r1/codeforces format
        self.test_item = {
            "id": "1A",
            "title": "Theatre Square",
            "description": "Theatre Square has length n and width m. Find the minimum number of flagstones needed to pave the Square.",
            "input_format": "The input contains three integers n, m, and a (1 ≤ n, m, a ≤ 10^9)",
            "output_format": "Output the minimum number of flagstones needed.",
            "difficulty": 1000,
            "tags": ["math", "implementation"],
            "examples": [
                {"input": "6 6 4", "output": "4"},
                {"input": "1 1 1", "output": "1"},
            ],
        }

        # Create a higher difficulty item
        self.high_difficulty_item = {
            "id": "158D",
            "title": "Ice Sculptures",
            "description": "Make ice sculptures...",
            "input_format": "Input specs...",
            "output_format": "Output specs...",
            "difficulty": 2000,  # Outside the range in self.config
            "tags": ["implementation", "math"],
            "examples": [{"input": "Sample input", "output": "Sample output"}],
        }

        # Create item with different tags
        self.different_tags_item = {
            "id": "101A",
            "title": "Game",
            "description": "Play a game...",
            "input_format": "Input specs...",
            "output_format": "Output specs...",
            "difficulty": 1200,
            "tags": ["dp", "games"],  # No "implementation" tag
            "examples": [{"input": "Sample input", "output": "Sample output"}],
        }

    def test_init_with_config(self) -> None:
        """CodeForcesPrepper should initialize with the provided config."""
        # Assert
        self.assertEqual(self.config, self.configured_prepper._config)
        self.assertEqual(
            self.config.difficulty_range, self.configured_prepper.difficulty_range
        )
        self.assertEqual(self.config.tags, self.configured_prepper.tags)
        self.assertEqual(self.config.limit, self.configured_prepper.limit)
        self.assertEqual(0, self.configured_prepper.processed_count)

    def test_init_without_config(self) -> None:
        """CodeForcesPrepper should create a default config when none is provided."""
        # Assert
        self.assertIsInstance(self.default_prepper._config, CodeForcesConfig)
        self.assertIsNone(self.default_prepper.difficulty_range)
        self.assertIsNone(self.default_prepper.tags)
        self.assertIsNone(self.default_prepper.limit)

    def test_get_required_keys(self) -> None:
        """get_required_keys() should return the expected list of keys."""
        # Arrange & Act
        required_keys = self.default_prepper.get_required_keys()

        # Assert
        expected_keys = [
            "id",
            "title",
            "description",
            "input_format",
            "output_format",
            "examples",
        ]
        self.assertEqual(set(expected_keys), set(required_keys))

    def test_create_dataset_entries_basic(self) -> None:
        """create_dataset_entries() should process items correctly."""
        # Arrange & Act
        entries = self.default_prepper.create_dataset_entries(item=self.test_item)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]

        # Check query is formatted properly
        self.assertTrue(entry.query.startswith(f"# {self.test_item['title']}"))
        self.assertIn(self.test_item["description"], entry.query)
        self.assertIn("## Input Specification", entry.query)
        self.assertIn("## Output Specification", entry.query)

        # Check empty choices (code completion has no choices)
        self.assertEqual({}, entry.choices)

        # Check metadata
        self.assertEqual(self.test_item["id"], entry.metadata["problem_id"])
        self.assertEqual(self.test_item["title"], entry.metadata["name"])
        self.assertEqual(self.test_item["difficulty"], entry.metadata["difficulty"])
        self.assertEqual(self.test_item["tags"], entry.metadata["tags"])
        self.assertEqual(self.test_item["examples"], entry.metadata["test_cases"])
        self.assertEqual("code_completion", entry.metadata["task_type"])
        self.assertEqual("codeforces", entry.metadata["dataset"])

        # Check processed_count is incremented
        self.assertEqual(1, self.default_prepper.processed_count)

    def test_create_dataset_entries_with_difficulty_filter(self) -> None:
        """create_dataset_entries() should filter by difficulty range."""
        # Act - Item within difficulty range should pass
        valid_entries = self.configured_prepper.create_dataset_entries(
            item=self.test_item
        )
        # Act - Item outside difficulty range should be filtered out
        filtered_entries = self.configured_prepper.create_dataset_entries(
            item=self.high_difficulty_item
        )

        # Assert
        self.assertEqual(1, len(valid_entries))
        self.assertEqual(0, len(filtered_entries))
        self.assertEqual(
            1, self.configured_prepper.processed_count
        )  # Only one item processed

    def test_create_dataset_entries_with_tags_filter(self) -> None:
        """create_dataset_entries() should filter by tags."""
        # Item with "implementation" tag should pass
        valid_entries = self.configured_prepper.create_dataset_entries(
            item=self.test_item
        )
        # Item without "implementation" tag should be filtered out
        filtered_entries = self.configured_prepper.create_dataset_entries(
            item=self.different_tags_item
        )

        # Assert
        self.assertEqual(1, len(valid_entries))
        self.assertEqual(0, len(filtered_entries))

    def test_create_dataset_entries_with_limit(self) -> None:
        """create_dataset_entries() should respect the limit."""
        # Arrange
        prepper = CodeForcesPrepper(config=CodeForcesConfig(limit=2))

        # Act - Process 3 valid items, but limit is 2
        entries1 = prepper.create_dataset_entries(item=self.test_item)  # Should process
        entries2 = prepper.create_dataset_entries(item=self.test_item)  # Should process
        entries3 = prepper.create_dataset_entries(
            item=self.test_item
        )  # Should be rejected due to limit

        # Assert
        self.assertEqual(1, len(entries1))
        self.assertEqual(1, len(entries2))
        self.assertEqual(0, len(entries3))
        self.assertEqual(2, prepper.processed_count)

    def test_create_dataset_entries_missing_fields(self) -> None:
        """create_dataset_entries() should handle missing optional fields gracefully."""
        # Arrange
        minimal_item = {
            "id": "101B",
            "title": "Minimal Problem",
            "description": "Solve this problem.",
            "examples": [{"input": "1", "output": "1"}],
        }

        # Act
        entries = self.default_prepper.create_dataset_entries(item=minimal_item)

        # Assert
        self.assertEqual(1, len(entries))
        entry = entries[0]
        self.assertEqual("", entry.metadata["input_specification"])
        self.assertEqual("", entry.metadata["output_specification"])
        self.assertEqual(0, entry.metadata["difficulty"])
        self.assertEqual([], entry.metadata["tags"])


if __name__ == "__main__":
    unittest.main()
