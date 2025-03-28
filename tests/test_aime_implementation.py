"""Test script for AIME dataset implementation.

This script verifies that the AIME dataset implementation works correctly.
"""

import unittest
from typing import List, Dict, Any

from ember.core.utils.data.base.models import DatasetEntry
from ember.core.utils.data.datasets_registry.aime import AIMEConfig, AIMEPrepper
from ember.core.utils.eval.numeric_answer import AIMEAnswerEvaluator


class TestAIMEImplementation(unittest.TestCase):
    """Test the AIME dataset implementation."""

    def test_aime_prepper(self):
        """Test that the AIME prepper works correctly."""
        # Create the prepper
        prepper = AIMEPrepper()

        # Check required keys
        required_keys = prepper.get_required_keys()
        self.assertIn("ID", required_keys)
        self.assertIn("Problem", required_keys)
        self.assertIn("Answer", required_keys)

        # Create a sample AIME problem
        item = {
            "ID": "2024-I-5",
            "Problem": "Find the sum of all positive integers n such that n^2 + 3n + 5 is divisible by 7.",
            "Answer": "42",
            "Solution": "Sample solution text",
        }

        # Test creating dataset entries
        entries = prepper.create_dataset_entries(item=item)
        self.assertEqual(len(entries), 1)

        entry = entries[0]
        self.assertEqual(entry.query, item["Problem"])
        self.assertEqual(entry.choices, {})  # No choices for short answer
        self.assertEqual(entry.metadata["correct_answer"], "42")
        self.assertEqual(entry.metadata["problem_id"], "2024-I-5")
        self.assertEqual(entry.metadata["solution"], "Sample solution text")

    def test_aime_answer_evaluator(self):
        """Test that the AIME answer evaluator works correctly."""
        evaluator = AIMEAnswerEvaluator()

        # Test correct answer
        result = evaluator.evaluate(
            "After solving the problem, I found the answer to be 42.", "42"
        )
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)

        # Test incorrect answer
        result = evaluator.evaluate("The answer is 43.", "42")
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)

        # Test multiple numbers in response
        result = evaluator.evaluate(
            "I first thought it was 10, then 25, but finally got 42.", "42"
        )
        self.assertTrue(result.is_correct)

        # Test invalid reference answer
        result = evaluator.evaluate("The answer is 42.", "1000")  # Out of AIME range
        self.assertFalse(result.is_correct)


if __name__ == "__main__":
    unittest.main()
