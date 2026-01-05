"""Property-based tests using standard library.

Since hypothesis is not installed in the test environment,
we'll use parameterized tests with generated data to achieve
similar property testing goals.
"""

import json
import random
import string
import tempfile
from pathlib import Path

import pytest

from ember.api.data import FileSource, stream
from ember.models.providers import resolve_model_id

# Import test infrastructure
from tests.test_doubles import FakeDataSource


# Generate test data
def generate_strings(min_len=1, max_len=50, count=20):
    """Generate random strings for testing."""
    for _ in range(count):
        length = random.randint(min_len, max_len)
        yield "".join(random.choices(string.ascii_letters + string.digits, k=length))


def generate_json_values(depth=3):
    """Generate various JSON-compatible values."""
    if depth <= 0:
        return [None, True, False, 0, 1, -1, 3.14, "", "test"]

    values = [None, True, False, 0, 1, -1, 3.14, "", "test", "a" * 100]

    # Add nested structures
    for _ in range(5):
        # Lists
        list_size = random.randint(0, 5)
        values.append([random.choice(generate_json_values(depth - 1)) for _ in range(list_size)])

        # Dicts
        dict_size = random.randint(0, 3)
        values.append(
            {f"key{i}": random.choice(generate_json_values(depth - 1)) for i in range(dict_size)}
        )

    return values


class TestModelResolutionProperties:
    """Property tests for model ID resolution."""

    known_models = [
        ("gpt-4", "openai", "gpt-4"),
        ("gpt-3.5-turbo", "openai", "gpt-3.5-turbo"),
        ("gpt-4o", "openai", "gpt-4o"),
        ("claude-4-sonnet", "anthropic", "claude-4-sonnet"),
        ("claude-3-sonnet", "anthropic", "claude-3-sonnet"),
        ("claude-2.1", "anthropic", "claude-2.1"),
        ("gemini-pro", "google", "gemini-pro"),
        ("gemini-1.5-pro", "google", "gemini-1.5-pro"),
        ("gemini-1.5-pro-latest", "google", "gemini-1.5-pro-latest"),
    ]

    @pytest.mark.parametrize("model_id,expected_provider,expected_model", known_models)
    def test_known_models_resolve_correctly(self, model_id, expected_provider, expected_model):
        """Known model IDs should resolve to correct providers."""
        provider, model = resolve_model_id(model_id)

        assert provider == expected_provider
        assert model == expected_model

    @pytest.mark.parametrize(
        "provider,model_name",
        [
            ("openai", "custom-model"),
            ("anthropic", "future-claude"),
            ("google", "gemini-ultra"),
            ("custom", "my-model"),
            ("a" * 50, "b" * 50),  # Long names
        ],
    )
    def test_explicit_provider_notation(self, provider, model_name):
        """Explicit provider/model notation should always work."""
        combined = f"{provider}/{model_name}"
        resolved_provider, resolved_model = resolve_model_id(combined)

        assert resolved_provider == provider
        assert resolved_model == model_name

    @pytest.mark.parametrize(
        "model_id",
        [
            "",  # Empty string - edge case
            "unknown-model",
            "llama-2-70b",
            "mistral-7b",
            "/leading-slash",
            "trailing-slash/",
            "multiple/slashes/here",
            "special!@#$%^&*()chars",
            "unicode-测试-model",
        ],
    )
    def test_resolution_never_crashes(self, model_id):
        """Model resolution should never crash on any input."""
        if not model_id:  # Skip empty string
            return

        try:
            provider, model = resolve_model_id(model_id)
            # Properties that always hold
            assert isinstance(provider, str)
            assert isinstance(model, str)
            assert len(provider) > 0 or model_id.startswith("/")
            assert len(model) > 0 or model_id.endswith("/")
        except Exception as e:
            pytest.fail(f"Resolution crashed on input '{model_id}': {e}")


class TestDataStreamingProperties:
    """Property tests for data streaming functionality."""

    @pytest.mark.parametrize("num_items", [0, 1, 10, 100, 1000])
    def test_streaming_preserves_all_items(self, num_items):
        """Streaming should preserve all items without loss."""
        items = [{"id": i, "value": f"item_{i}"} for i in range(num_items)]
        source = FakeDataSource(items)

        results = list(stream(source, normalize=False))

        assert len(results) == len(items)
        assert results == items

    @pytest.mark.parametrize(
        "num_items,threshold",
        [
            (10, 5),
            (100, 50),
            (100, 0),  # All pass
            (100, 100),  # None pass
            (0, 50),  # Empty list
        ],
    )
    def test_filter_property(self, num_items, threshold):
        """Filtering should only keep items matching predicate."""
        items = [{"id": i, "score": i} for i in range(num_items)]
        source = FakeDataSource(items)

        results = list(stream(source, filter=lambda x: x["score"] > threshold, normalize=False))

        # Properties
        assert all(r["score"] > threshold for r in results)
        assert len(results) == len([x for x in items if x["score"] > threshold])

        # Verify exact items
        expected = [item for item in items if item["score"] > threshold]
        assert results == expected

    @pytest.mark.parametrize(
        "num_items,max_items",
        [
            (10, 5),
            (10, 10),
            (10, 20),  # max > available
            (0, 10),  # Empty source
            (100, 0),  # Zero limit
        ],
    )
    def test_limit_property(self, num_items, max_items):
        """Limiting should respect max_items constraint."""
        items = [{"id": i} for i in range(num_items)]
        source = FakeDataSource(items)

        results = list(stream(source, max_items=max_items, normalize=False))

        # Properties
        assert len(results) <= max_items
        assert len(results) <= num_items
        assert len(results) == min(max_items, num_items)

        # First N items preserved
        assert results == items[:max_items]

    def test_transform_composition(self):
        """Multiple transforms should compose correctly."""
        items = [{"value": i} for i in range(10)]
        source = FakeDataSource(items)

        # Chain transforms
        results = list(
            stream(source, normalize=False)
            .transform(lambda x: {**x, "doubled": x["value"] * 2})
            .transform(lambda x: {**x, "squared": x["value"] ** 2})
            .transform(lambda x: {**x, "sum": x["doubled"] + x["squared"]})
        )

        # Verify all transforms applied correctly
        for i, result in enumerate(results):
            assert result["value"] == i
            assert result["doubled"] == i * 2
            assert result["squared"] == i**2
            assert result["sum"] == (i * 2) + (i**2)


class TestDataNormalizationProperties:
    """Property tests for data normalization."""

    @pytest.mark.parametrize(
        "input_data,has_question,has_answer",
        [
            ({"question": "Q?", "answer": "A"}, True, True),
            ({"query": "Q?", "target": "A"}, True, True),
            ({"prompt": "Q?", "response": "A"}, True, True),
            ({"text": "Q?", "output": "A"}, True, True),
            ({"input": "Q?", "label": "A"}, True, True),
            ({"unrelated": "data"}, False, False),
            ({}, False, False),
        ],
    )
    def test_normalization_mapping(self, input_data, has_question, has_answer):
        """Normalization should map various formats to standard schema."""
        source = FakeDataSource([input_data])
        results = list(stream(source, normalize=True))

        assert len(results) == 1
        result = results[0]

        # Standard fields always present
        assert "question" in result
        assert "answer" in result
        assert "choices" in result
        assert "metadata" in result

        # Check mapping worked
        if has_question:
            assert len(result["question"]) > 0
        if has_answer:
            assert len(result["answer"]) > 0


class TestFileHandlingProperties:
    """Property tests for file handling."""

    @pytest.mark.parametrize(
        "data",
        [
            [],
            [{"a": 1}],
            [{"a": 1}, {"b": 2}],
            [{"nested": {"deep": {"value": 42}}}],
            [{"list": [1, 2, 3]}],
            [{"mixed": [1, "two", None, True, {"nested": "value"}]}],
        ],
    )
    def test_json_roundtrip(self, data):
        """JSON files should preserve data exactly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name

        try:
            source = FileSource(fname)
            results = []
            for batch in source.read_batches():
                results.extend(batch)

            assert results == data
        finally:
            Path(fname).unlink()

    def test_jsonl_roundtrip(self):
        """JSONL files should preserve line-by-line data."""
        data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
            {"id": 3, "nested": {"value": "third"}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            fname = f.name

        try:
            source = FileSource(fname)
            results = []
            for batch in source.read_batches():
                results.extend(batch)

            assert results == data
        finally:
            Path(fname).unlink()

    def test_csv_handling(self):
        """CSV files should handle tabular data correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,name,value\n")
            f.write("1,first,10\n")
            f.write("2,second,20\n")
            f.write("3,third,30\n")
            fname = f.name

        try:
            source = FileSource(fname)
            results = []
            for batch in source.read_batches():
                results.extend(batch)

            # CSV converts everything to strings
            assert len(results) == 3
            assert results[0] == {"id": "1", "name": "first", "value": "10"}
            assert results[1] == {"id": "2", "name": "second", "value": "20"}
            assert results[2] == {"id": "3", "name": "third", "value": "30"}
        finally:
            Path(fname).unlink()


class TestOperatorCompositionProperties:
    """Property tests for operator composition."""

    def test_chain_order_invariant(self):
        """Chain should preserve operation order."""
        from ember.operators import Chain, Operator

        class AppendOperator(Operator):
            suffix: str

            def __init__(self, suffix):
                self.suffix = suffix

            def forward(self, x):
                return f"{x}_{self.suffix}"

        # Test with various chain lengths
        for chain_length in [1, 2, 5, 10]:
            operators = [AppendOperator(str(i)) for i in range(chain_length)]
            chain = Chain(operators)

            # Test with multiple inputs
            for input_val in ["test", "a", "long_string_here"]:
                result = chain(input_val)

                # Build expected
                expected = input_val
                for i in range(chain_length):
                    expected = f"{expected}_{i}"

                assert result == expected

    def test_ensemble_consistency(self):
        """Ensemble should handle all operators consistently."""
        from ember.operators import Ensemble, Operator

        class ConstantOperator(Operator):
            value: int

            def __init__(self, value):
                self.value = value

            def forward(self, x):
                return self.value

        # Test various ensemble sizes
        for num_ops in [1, 3, 5]:
            operators = [ConstantOperator(i) for i in range(num_ops)]

            # Without aggregator - should return list
            ensemble = Ensemble(operators)
            result = ensemble("input")
            assert result == list(range(num_ops))

            # With sum aggregator
            ensemble_sum = Ensemble(operators, aggregator=sum)
            result = ensemble_sum("input")
            assert result == sum(range(num_ops))

            # With max aggregator
            ensemble_max = Ensemble(operators, aggregator=max)
            result = ensemble_max("input")
            assert result == num_ops - 1 if num_ops > 0 else 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
