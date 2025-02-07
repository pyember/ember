"""Integration tests for ember components."""

import os
import logging
import unittest
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from ember.core.non_graph import NoNGraph
from ember.registry.model_registry.model_registry import (
    ModelCost,
    ModelInfo,
    ModelRegistry,
    OpenAIModel,
    ProviderInfo,
    RateLimit,
)
from ember.registry.operators.operator_base import OperatorContext
from ember.registry.operators.operator_registry import (
    EnsembleOperator,
    GetAnswerOperator,
    MostCommonOperator,
    OperatorFactory,
    OperatorRegistry,
)


class ExampleNoN(NoNGraph):
    """Example NoN graph for testing purposes."""

    def __init__(self):
        """Initialize the example NoN graph with a basic structure."""
        super().__init__()

        # Defining a basic graph structure
        self.add_node(
            "node1",
            OperatorFactory.create("E", [{"model_name": "gpt-4-turbo", "count": 3}]),
        )
        self.add_node(
            "node2",
            OperatorFactory.create("E", [{"model_name": "gpt-4-turbo", "count": 1}]),
        )
        self.add_node(
            "node3",
            OperatorFactory.create("MC", [{"model_name": "gpt-4-turbo", "count": 1}]),
            inputs=["node1", "node2"],
        )


class MockPromptGenerator:
    """Mock class for prompt generation."""

    def generate_prompt(self, *args: Any, **kwargs: Any) -> str:
        """Generate a mock prompt."""
        return "Mocked prompt"


class MockOpenAIResponse:
    """Mock class for OpenAI API responses."""

    def __init__(self, content: str):
        """Initialize the mock response with given content."""
        self.choices = [Mock(message=Mock(content=content))]
        self.usage = Mock(total_tokens=100, prompt_tokens=50, completion_tokens=50)


class TestIntegration(unittest.TestCase):
    """Integration tests for ember components."""

    def setUp(self):
        """Set up the test environment."""
        # Setting up environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Creating ModelInfo
        provider_info = ProviderInfo(name="OpenAI", default_api_key=self.api_key)
        model_cost = ModelCost(
            input_cost_per_thousand=0.03, output_cost_per_thousand=0.06
        )
        rate_limit = RateLimit(tokens_per_minute=90000, requests_per_minute=3500)

        self.model_info = ModelInfo(
            model_id="gpt-4",
            model_name="gpt-4",
            cost=model_cost,
            rate_limit=rate_limit,
            provider=provider_info,
            api_key=self.api_key,
        )

        # Setting up ModelRegistry
        self.model_registry = ModelRegistry()
        self.model_registry.register_model(
            self.model_info, OpenAIModel(self.model_info)
        )

        # Registering operators
        # Ensure that PromptNames used here are valid
        OperatorRegistry.register(EnsembleOperator.get_metadata(), EnsembleOperator)
        OperatorRegistry.register(MostCommonOperator.get_metadata(), MostCommonOperator)
        OperatorRegistry.register(GetAnswerOperator.get_metadata(), GetAnswerOperator)

    @patch("ember.registry.operator_registry.LMOperatorUnit.forward")
    def test_ensemble_operator(self, mock_unit_forward):
        """Testing the EnsembleOperator."""
        mock_unit_forward.side_effect = ["Paris", "Paris", "London"]

        ensemble_op = EnsembleOperator(num_units=3, model_name="gpt-4o")

        input_data = OperatorContext(
            query="What is the capital of France?",
            context="You are a helpful assistant.",
            choices={"A": "London", "B": "Paris", "C": "Berlin"},
        )

        result = ensemble_op.forward(input_data)
        print("result: ", result)

        self.assertEqual(len(result.final_answer), 3)
        self.assertIn("Paris", result.final_answer)
        self.assertIn("London", result.final_answer)

    @patch("ember.registry.operator_registry.MostCommonOperator.forward")
    def test_most_common_operator(self, mock_forward):
        """Testing the MostCommonOperator."""
        mock_forward.return_value = "Paris"

        mc_op = OperatorFactory.create(
            "MC", [{"model_name": "gpt-4-turbo", "count": 1}]
        )

        responses = ["Paris", "Paris", "London"]
        question = "What is the capital of France?"
        choices = {"A": "London", "B": "Paris", "C": "Berlin"}

        input_data = OperatorContext(
            query=question, choices=choices, responses=responses
        )

        result = mc_op.forward(input_data)

        self.assertEqual(result, "Paris")

    @patch("ember.registry.operator_registry.EnsembleOperator.forward")
    @patch("ember.registry.operator_registry.MostCommonOperator.forward")
    def test_full_integration(self, mock_mc_forward, mock_ensemble_forward):
        """Testing full integration of operators in a graph."""
        mock_ensemble_forward.return_value = ["Paris", "London", "Paris"]
        mock_mc_forward.return_value = "Paris"

        graph_definition = [["3:E:gpt-4:1.0"], "1:MC"]
        non_graph = NoNGraph().parse_from_list(graph_definition)

        input_data = OperatorContext(
            query="What is the capital of France?",
            context="You are a helpful assistant.",
            choices={"A": "London", "B": "Paris", "C": "Berlin"},
        )

        result = non_graph.forward(input_data)

        self.assertEqual(result.final_answer, "Paris")

    @patch("ember.registry.operator_registry.EnsembleOperator.forward")
    @patch("ember.registry.operator_registry.MostCommonOperator.forward")
    def test_graph_creation_and_execution(self, mock_mc_forward, mock_ensemble_forward):
        """Testing graph creation and execution."""
        mock_ensemble_forward.return_value = ["Response 1", "Response 2", "Response 3"]
        mock_mc_forward.return_value = "Response 2"

        graph_definition = [["3:E:gpt-4:1.0"], "1:MC"]
        non_graph = NoNGraph().parse_from_list(graph_definition)

        input_data = OperatorContext(
            query="What is the capital of France?",
            context="You are a helpful assistant.",
            choices={"A": "London", "B": "Paris", "C": "Berlin"},
        )

        result = non_graph.forward(input_data)

        self.assertIsNotNone(result)
        self.assertEqual(result.final_answer, "Response 2")

    def test_error_handling(self):
        """Testing error handling for invalid inputs."""
        test_cases = [
            {
                "description": "invalid operator code",
                "callable": lambda: OperatorFactory.create(
                    "InvalidCode", [{"model_name": "gpt-4", "count": 1}]
                ),
                "expected_message": "Unknown operator code: InvalidCode",
            },
            {
                "description": "invalid graph definition",
                "callable": lambda: NoNGraph().parse_from_list([["InvalidOperator"]]),
                "expected_message": "Invalid operation",
            },
        ]

        for case in test_cases:
            with self.subTest(case["description"]):
                with self.assertRaises(ValueError) as context:
                    case["callable"]()
                self.assertIn(case["expected_message"], str(context.exception))

    @patch("openai.OpenAI")
    def test_openai_model_integration(self, mock_openai):
        """Testing OpenAI model integration."""
        mock_openai.return_value.chat.completions.create.return_value = (
            MockOpenAIResponse("Test response")
        )

        model = OpenAIModel(self.model_info)
        self.assertIsInstance(model, OpenAIModel)

        input_data = OperatorContext(
            query="Test question. Return 'Test response' as your response.",
            context="Test context",
        )
        raw_output = model.forward(input_data, process_output=False, max_tokens=100)
        processed_output = model._process_output(raw_output)

        self.assertEqual(processed_output, "Test response")
        usage = model.calculate_usage(raw_output)
        self.assertLessEqual(usage["total_tokens"], 100)

    # def test_example_non(self):
    #     """Testing example NoN graph."""
    #     from ember.registry.operator_registry import EnsembleOperator, MostCommonOperator, OperatorFactory

    #     graph_definition = [
    #         ['2:E:gpt-4o:1.0'],
    #         '1:MC'
    #     ]

    #     example_non = NoNGraph().parse_from_list(graph_definition)

    #     input_data = OperatorContext(
    #         question="What is the capital of France?",
    #         context="You are a helpful assistant.",
    #         choices={"A": "London", "B": "Paris", "C": "Berlin"}
    #     )

    #     result = example_non.forward(input_data)

    #     self.assertIsNotNone(result)
    #     self.assertIn(result.final_answer, ["B"])
    #     self.assertEqual(len(example_non.execution_order), 2)
    #     self.assertEqual(
    #         example_non.node_inputs["node_1"], ["node_0"]
    #     )


if __name__ == "__main__":
    unittest.main()
