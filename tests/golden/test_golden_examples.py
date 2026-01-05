"""Example tests using golden file management.

These tests demonstrate how to use golden files for:
- Model response comparisons
- Configuration output validation
- Data transformation results
"""

import pytest

from ember.api.data import stream
from ember.models.providers import resolve_model_id
from tests.golden import GoldenFileManager, compare_golden_json, compare_golden_text
from tests.test_doubles import FakeDataSource, FakeProvider


class TestModelResponseGolden:
    """Test model responses against golden files."""

    def test_provider_response_format(self):
        """Test that provider responses match expected format."""
        # Create fake provider with deterministic responses
        provider = FakeProvider(
            responses={
                "test prompt": "This is a test response",
                "math problem": "2 + 2 = 4",
            }
        )

        # Get responses
        responses = []
        for prompt in ["test prompt", "math problem", "unknown"]:
            response = provider.complete(prompt, "test-model")
            responses.append(
                {
                    "prompt": prompt,
                    "response": response.data,
                    "model": response.model_id,
                    "tokens": response.usage.total_tokens,
                }
            )

        # Compare with golden file
        compare_golden_json("provider_response_format", responses)

    def test_model_resolution_golden(self):
        """Test model ID resolution against golden outputs."""
        test_cases = [
            "gpt-4",
            "claude-4-sonnet",
            "gemini-2.5-pro",
            "gemini-1.5-pro-latest",
            "openai/gpt-4",
            "custom/model",
            "unknown-model",
            "special/chars/in/model",
        ]

        results = []
        for model_id in test_cases:
            provider, model = resolve_model_id(model_id)
            results.append({"input": model_id, "provider": provider, "model": model})

        compare_golden_json("model_resolution_cases", results)


class TestDataTransformationGolden:
    """Test data transformations against golden files."""

    def test_data_normalization_golden(self):
        """Test that data normalization produces expected outputs."""
        # Various input formats that should normalize
        test_data = [
            {"question": "Q1", "answer": "A1"},
            {"query": "Q2", "target": "A2"},
            {"prompt": "Q3", "response": "A3"},
            {"text": "Q4", "output": "A4"},
            {"input": "Q5", "label": "A5"},
            {"unrelated": "data", "fields": "here"},
            {},
        ]

        source = FakeDataSource(test_data)
        normalized = list(stream(source, normalize=True))

        # Extract relevant fields for golden comparison
        results = []
        for item in normalized:
            results.append(
                {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "has_choices": len(item.get("choices", {})) > 0,
                    "metadata_keys": sorted(item.get("metadata", {}).keys()),
                }
            )

        compare_golden_json("data_normalization", results)

    def test_streaming_pipeline_golden(self):
        """Test complex streaming pipeline against golden output."""
        # Create test data
        items = [
            {"id": i, "value": i * 10, "category": "even" if i % 2 == 0 else "odd"}
            for i in range(10)
        ]

        source = FakeDataSource(items)

        # Apply pipeline
        results = list(
            stream(source, normalize=False)
            .filter(lambda x: x["value"] > 20)
            .transform(lambda x: {**x, "squared": x["value"] ** 2})
            .transform(lambda x: {**x, "label": f"{x['category']}_{x['id']}"})
            .limit(3)
        )

        compare_golden_json("streaming_pipeline", results)


class TestTextOutputGolden:
    """Test text outputs against golden files."""

    def test_formatted_output_golden(self):
        """Test formatted text output."""
        # Generate a formatted report
        data = [
            {"name": "Alice", "score": 95, "grade": "A"},
            {"name": "Bob", "score": 82, "grade": "B"},
            {"name": "Charlie", "score": 78, "grade": "C"},
        ]

        report = "Student Grade Report\n"
        report += "=" * 40 + "\n"
        report += f"{'Name':<15} {'Score':>10} {'Grade':>10}\n"
        report += "-" * 40 + "\n"

        for student in data:
            report += f"{student['name']:<15} {student['score']:>10} {student['grade']:>10}\n"

        report += "-" * 40 + "\n"
        report += f"Average Score: {sum(s['score'] for s in data) / len(data):.1f}\n"

        compare_golden_text("formatted_report", report)


class TestGoldenFileManager:
    """Test the golden file manager itself."""

    def test_manager_update_mode(self, tmp_path, monkeypatch):
        """Test update mode functionality."""
        manager = GoldenFileManager(tmp_path)

        # Initially not in update mode
        assert not manager.update_mode

        # Test with update mode
        monkeypatch.setenv("UPDATE_GOLDEN_FILES", "true")
        manager2 = GoldenFileManager(tmp_path)
        assert manager2.update_mode

        # Should save when in update mode
        manager2.compare("test", {"data": "value"})
        assert (tmp_path / "test.json").exists()

    def test_manager_comparison(self, tmp_path):
        """Test comparison functionality."""
        manager = GoldenFileManager(tmp_path)

        # First run creates file
        manager.compare("first_run", {"key": "value"})
        assert (tmp_path / "first_run.json").exists()

        # Same data passes
        manager.compare("first_run", {"key": "value"})

        # Different data fails with nice error
        with pytest.raises(AssertionError) as exc_info:
            manager.compare("first_run", {"key": "different"})

        error_msg = str(exc_info.value)
        assert "Output doesn't match golden file" in error_msg
        assert "UPDATE_GOLDEN_FILES=true" in error_msg
        assert "@@" in error_msg  # Diff marker


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
