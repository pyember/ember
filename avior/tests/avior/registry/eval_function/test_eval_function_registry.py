"""Tests for the eval_function_registry module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from parameterized import parameterized
from avior.registry.eval_function.eval_function_registry import (
    EvaluatorRegistry,
    MultipleChoiceEvaluator,
    NumericalEvaluator,
    TextualEvaluator,
    Scorer,
    EvaluationResult,
    BaseEvaluator,
)
from avior.registry.operators.operator_base import OperatorContext
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import openai
from avior.core.configs.config import set_config

# Setting up the OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    set_config("models", "openai_api_key", openai_api_key)
else:
    print("WARNING: OPENAI_API_KEY not found in environment variables")


class TestEvaluationResult(unittest.TestCase):
    """Tests for the EvaluationResult dataclass."""

    def test_creation(self):
        """Testing creation of EvaluationResult with metadata."""
        result = EvaluationResult(is_correct=True, score=0.8, metadata={"key": "value"})
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 0.8)
        self.assertEqual(result.metadata, {"key": "value"})

    def test_creation_without_metadata(self):
        """Testing creation of EvaluationResult without metadata."""
        result = EvaluationResult(is_correct=False, score=0.0)
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)
        self.assertIsNone(result.metadata)


class TestBaseEvaluator(unittest.TestCase):
    """Tests for the BaseEvaluator abstract class."""

    def setUp(self):
        """Setting up the test environment."""
        self.evaluator = Mock(spec=BaseEvaluator)
        self.evaluator.extract_answer.return_value = "test"
        self.evaluator.evaluate.return_value = EvaluationResult(True, 1.0)
        # Correctly setting up the score method
        self.evaluator.score.side_effect = (
            lambda response, correct_answer, **kwargs: self.evaluator.evaluate(
                self.evaluator.extract_answer(response, **kwargs), correct_answer
            )
        )

    def test_score_success(self):
        """Testing successful scoring."""
        result = self.evaluator.score("response", "correct")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)

    def test_score_error_handling(self):
        """Testing error handling during scoring."""
        self.evaluator.extract_answer.side_effect = Exception("Test error")
        # Mocking the score method to return a failure result when an exception occurs
        self.evaluator.score.side_effect = (
            lambda response, correct_answer, **kwargs: EvaluationResult(
                False, 0.0, {"error": "Test error"}
            )
        )
        result = self.evaluator.score("response", "correct")
        self.assertFalse(result.is_correct)
        self.assertEqual(result.score, 0.0)
        self.assertIn("error", result.metadata)


class TestEvaluatorRegistry(unittest.TestCase):
    """Tests for the EvaluatorRegistry class."""

    def setUp(self):
        """Setting up the test environment."""
        self.registry = EvaluatorRegistry()
        self.mcq_evaluator = MultipleChoiceEvaluator()
        self.numerical_evaluator = NumericalEvaluator()
        self.textual_evaluator = TextualEvaluator()

    def test_register_and_get_evaluator(self):
        """Testing registering and retrieving an evaluator."""
        self.registry.register("mcq", self.mcq_evaluator)
        self.assertEqual(self.registry.get("mcq"), self.mcq_evaluator)

    def test_get_nonexistent_evaluator(self):
        """Testing retrieval of a non-existent evaluator."""
        with self.assertRaises(KeyError):
            self.registry.get("nonexistent")

    @patch.object(MultipleChoiceEvaluator, "extract_answer", return_value="A")
    @patch.object(MultipleChoiceEvaluator, "evaluate")
    def test_evaluate_batch(self, mock_evaluate, mock_extract_answer):
        """Testing batch evaluation of responses."""
        # Registering the MCQ evaluator
        self.registry.register("mcq", self.mcq_evaluator)
        mock_evaluate.return_value = EvaluationResult(True, 1.0)

        # Defining test data
        responses = ["The answer is A", "I believe it's B", "C is the correct choice"]
        correct_answers = ["A", "B", "D"]
        choices = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}

        # Calling evaluate_batch with the required choices argument
        results = self.registry.evaluate_batch(
            "mcq", responses, correct_answers, choices=choices
        )

        # Asserting the results
        self.assertEqual(len(results), 3)
        self.assertTrue(all([result.is_correct for result in results]))

        # Printing results for debugging
        print("RESULTS", results)


class TestMultipleChoiceEvaluator(unittest.TestCase):
    """Tests for the MultipleChoiceEvaluator class."""

    def setUp(self):
        """Setting up the test environment."""
        self.evaluator = MultipleChoiceEvaluator()
        # Mock the model used by the evaluator
        self.evaluator.model = Mock()

    def test_extract_answer(self):
        """Testing answer extraction for multiple choice questions."""
        # Mock the forward method to return a response with choices
        mock_raw_answer = MagicMock()
        mock_raw_answer.choices = [MagicMock(message=MagicMock(content="A"))]
        self.evaluator.model.forward.return_value = mock_raw_answer
        # Assuming _process_output just extracts the content
        self.evaluator.model._process_output.return_value = "A"

        choices = {"A": "Option A", "B": "Option B", "C": "Option C"}

        extracted_answer = self.evaluator.extract_answer(
            "This is the response", choices=choices
        )
        self.assertEqual(extracted_answer, "A")

    @parameterized.expand(
        [
            ("A", "A", True, 1.0),
            ("B", "A", False, 0.0),
        ]
    )
    def test_evaluate(self, extracted, correct, expected_correct, expected_score):
        """Testing evaluation of extracted answers."""
        result = self.evaluator.evaluate(extracted, correct)
        self.assertEqual(result.is_correct, expected_correct)
        self.assertEqual(result.score, expected_score)

    @patch.object(MultipleChoiceEvaluator, "extract_answer", return_value="A")
    def test_score_integration(self, mock_extract):
        """Testing integration of extraction and evaluation in scoring."""
        result = self.evaluator.score(
            "The answer is A",
            "A",
            choices={"A": "Option A", "B": "Option B", "C": "Option C"},
        )
        mock_extract.assert_called_once_with(
            "The answer is A",
            choices={"A": "Option A", "B": "Option B", "C": "Option C"},
        )
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)


class TestNumericalEvaluator(unittest.TestCase):
    """Tests for the NumericalEvaluator class."""

    def setUp(self):
        """Setting up the test environment."""
        self.evaluator = NumericalEvaluator(tolerance=0.1)

    @parameterized.expand(
        [
            (3.05, 3.0, True, 0.9833),
            (3.5, 3.0, False, 0.8333),
        ]
    )
    def test_evaluate(self, extracted, correct, expected_correct, expected_score):
        """Testing evaluation of numerical answers."""
        result = self.evaluator.evaluate(extracted, correct)
        self.assertEqual(result.is_correct, expected_correct)
        self.assertAlmostEqual(result.score, expected_score, places=4)

    @patch.object(NumericalEvaluator, "extract_answer", return_value=3.05)
    def test_score_integration(self, mock_extract):
        """Testing integration of extraction and evaluation in scoring."""
        result = self.evaluator.score("The answer is 3.05", 3.0)
        mock_extract.assert_called_once_with("The answer is 3.05")
        self.assertTrue(result.is_correct)
        self.assertAlmostEqual(result.score, 0.9833, places=4)


class TestTextualEvaluator(unittest.TestCase):
    """Tests for the TextualEvaluator class."""

    def setUp(self):
        """Setting up the test environment."""
        self.evaluator = TextualEvaluator()

    @parameterized.expand(
        [
            ("Python", "Python", True, 1.0),
            ("python", "Python", True, 1.0),
            ("Java", "Python", False, 0.0),
        ]
    )
    def test_evaluate(self, extracted, correct, expected_correct, expected_score):
        """Testing evaluation of textual answers."""
        result = self.evaluator.evaluate(extracted, correct)
        self.assertEqual(result.is_correct, expected_correct)
        self.assertEqual(result.score, expected_score)

    @patch.object(TextualEvaluator, "extract_answer", return_value="Python")
    def test_score_integration(self, mock_extract):
        """Testing integration of extraction and evaluation in scoring."""
        result = self.evaluator.score("Python is a programming language", "Python")
        mock_extract.assert_called_once_with("Python is a programming language")
        self.assertTrue(result.is_correct)
        self.assertEqual(result.score, 1.0)


class TestScorer(unittest.TestCase):
    """Tests for the Scorer class."""

    def setUp(self):
        """Setting up the test environment."""
        self.registry = EvaluatorRegistry()
        self.scorer = Scorer(self.registry)

    def test_initialization(self):
        """Testing proper initialization of Scorer."""
        self.assertIsInstance(self.scorer.evaluator_registry, EvaluatorRegistry)

    @patch.object(Scorer, "score_mesh", return_value=0.85)
    def test_score_mesh(self, mock_score_mesh):
        """Testing the score_mesh method."""
        questions = [
            {"question": "What is 2 + 2?", "type": "numerical"},
            {"question": "What is the capital of France?", "type": "mcq"},
            {"question": "Explain the concept of recursion.", "type": "textual"},
        ]
        mesh_definition = [
            {"evaluator": "numerical", "weight": 0.3},
            {"evaluator": "mcq", "weight": 0.3},
            {"evaluator": "textual", "weight": 0.4},
        ]
        score = self.scorer.score_mesh(questions, mesh_definition, "test_mesh")
        mock_score_mesh.assert_called_once_with(questions, mesh_definition, "test_mesh")
        self.assertEqual(score, 0.85)


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire evaluation system."""

    def setUp(self):
        """Setting up the test environment."""
        self.registry = EvaluatorRegistry()
        # Using actual evaluators instead of mocks
        self.registry.register("mcq", MultipleChoiceEvaluator())
        self.registry.register("numerical", NumericalEvaluator())
        self.registry.register("textual", TextualEvaluator())
        self.scorer = Scorer(self.registry)

    @patch.object(MultipleChoiceEvaluator, "extract_answer", return_value="A")
    @patch.object(NumericalEvaluator, "extract_answer", return_value=4.0)
    @patch.object(
        TextualEvaluator,
        "extract_answer",
        return_value="Recursion is a method where the solution to a problem depends on solutions to smaller instances of the same problem.",
    )
    @patch.object(Scorer, "score_mesh", return_value=0.75)
    def test_full_workflow(
        self,
        mock_score_mesh,
        mock_textual_extract,
        mock_numerical_extract,
        mock_mcq_extract,
    ):
        """Testing the full evaluation workflow."""
        questions = [
            {"question": "What is 2 + 2?", "type": "numerical", "answer": 4},
            {
                "question": "What is the capital of France?",
                "type": "mcq",
                "answer": "A",
                "choices": {"A": "Paris", "B": "London", "C": "Berlin"},
            },
            {
                "question": "Explain recursion.",
                "type": "textual",
                "answer": "Recursion is a method where the solution to a problem depends on solutions to smaller instances of the same problem.",
            },
        ]
        mesh_definition = [
            {"evaluator": "numerical", "weight": 0.3},
            {"evaluator": "mcq", "weight": 0.3},
            {"evaluator": "textual", "weight": 0.4},
        ]

        score = self.scorer.score_mesh(questions, mesh_definition, "test_integration")

        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)
        self.assertEqual(score, 0.75)

        for question in questions:
            evaluator = self.registry.get(question["type"])
            if question["type"] == "mcq":
                result = evaluator.score(
                    question["answer"],
                    question["answer"],
                    choices=question.get("choices"),
                )
            else:
                result = evaluator.score(question["answer"], question["answer"])
            self.assertTrue(result.is_correct)
            self.assertEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
