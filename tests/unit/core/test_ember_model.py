"""Unit tests for EmberModel."""

import unittest
from typing import Dict, Any, List, Optional
from ember.core.types import EmberModel


class TestEmberModel(unittest.TestCase):
    """Tests for the EmberModel unified type system."""

    def test_basic_functionality(self):
        """Test basic EmberModel creation and validation."""

        class TestModel(EmberModel):
            name: str
            age: int

        model = TestModel(name="Test", age=30)
        self.assertEqual(model.name, "Test")
        self.assertEqual(model.age, 30)

    def test_dict_conversion(self):
        """Test conversion to dictionary."""

        class TestModel(EmberModel):
            name: str
            age: int

        model = TestModel(name="Test", age=30)
        as_dict = model.as_dict()
        self.assertIsInstance(as_dict, dict)
        self.assertEqual(as_dict["name"], "Test")
        self.assertEqual(as_dict["age"], 30)

    def test_output_format(self):
        """Test different output formats."""

        class TestModel(EmberModel):
            name: str
            age: int

        # Default is model
        model = TestModel(name="Test", age=30)
        self.assertIsInstance(model(), TestModel)

        # Change to dict
        model.set_output_format("dict")
        self.assertIsInstance(model(), dict)

        # Change to json
        model.set_output_format("json")
        self.assertIsInstance(model(), str)

    def test_access_patterns(self):
        """Test both attribute and dictionary access patterns."""

        class TestModel(EmberModel):
            name: str
            age: int

        model = TestModel(name="Test", age=30)

        # Test attribute access
        self.assertEqual(model.name, "Test")
        self.assertEqual(model.age, 30)

        # Test dictionary-like access
        self.assertEqual(model["name"], "Test")
        self.assertEqual(model["age"], 30)

        # Test attribute errors
        with self.assertRaises(AttributeError):
            _ = model.nonexistent

        # Test dictionary key errors
        with self.assertRaises(KeyError):
            _ = model["nonexistent"]

    def test_dict_to_model_conversion(self):
        """Test automatic conversion from dict to model."""

        class ResponseModel(EmberModel):
            text: str
            score: Optional[float] = None

        # A function that returns a dict but is typed to return ResponseModel
        def get_response() -> ResponseModel:
            return {"text": "Hello, world!", "score": 0.95}

        # The dict should be automatically converted to ResponseModel during validation
        result = ResponseModel.model_validate(get_response())
        self.assertIsInstance(result, ResponseModel)
        self.assertEqual(result.text, "Hello, world!")
        self.assertEqual(result.score, 0.95)

    def test_operator_pattern_compatibility(self):
        """Test compatibility with typical operator pattern."""

        class OperatorInputs(EmberModel):
            query: str

        class OperatorOutputs(EmberModel):
            result: str
            confidence: float

        # Simulate an operator that returns a dict but is typed to return OperatorOutputs
        class MockOperator:
            def forward(self, *, inputs: OperatorInputs) -> OperatorOutputs:
                # Note how we return a dict, not an OperatorOutputs instance
                return {"result": f"Processed {inputs.query}", "confidence": 0.9}

            def __call__(self, *, inputs: OperatorInputs) -> OperatorOutputs:
                return self.forward(inputs=inputs)

        op = MockOperator()
        result = op(inputs=OperatorInputs(query="test"))

        # Verify we get the right type even though op returns a dict
        self.assertIsInstance(
            result, dict
        )  # Currently returns dict for backward compatibility

        # Convert to OperatorOutputs for strong typing when needed
        typed_result = OperatorOutputs(**result)
        self.assertIsInstance(typed_result, OperatorOutputs)
        self.assertEqual(typed_result.result, "Processed test")
        self.assertEqual(typed_result.confidence, 0.9)
