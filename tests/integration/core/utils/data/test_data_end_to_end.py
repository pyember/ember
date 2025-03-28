"""End-to-end data processing integration tests.

Tests the complete data processing pipeline from loading to transformation.
Uses minimal test doubles instead of actual implementations.
"""

import json
import os
import tempfile

import pytest

# Import minimal test doubles instead of actual implementations
from tests.helpers.data_minimal_doubles import MinimalDataService as DataService
from tests.helpers.data_minimal_doubles import MinimalFilter as Filter
from tests.helpers.data_minimal_doubles import MinimalJsonLoader as JsonLoader
from tests.helpers.data_minimal_doubles import MinimalRandomSampler as RandomSampler
from tests.helpers.data_minimal_doubles import MinimalSchemaValidator as SchemaValidator
from tests.helpers.data_minimal_doubles import MinimalShuffler as Shuffler
from tests.helpers.data_minimal_doubles import MinimalTextLoader as TextLoader
from tests.helpers.data_minimal_doubles import MinimalTokenizer as Tokenizer

# Mark all tests as integration tests
pytestmark = [pytest.mark.integration]


@pytest.fixture
def sample_json_data():
    """Create a temporary JSON file with sample data."""
    data = [
        {
            "id": 1,
            "question": "What is the capital of France?",
            "answer": "Paris",
            "difficulty": "easy",
        },
        {
            "id": 2,
            "question": "What is the capital of Germany?",
            "answer": "Berlin",
            "difficulty": "easy",
        },
        {
            "id": 3,
            "question": "What is the capital of Italy?",
            "answer": "Rome",
            "difficulty": "easy",
        },
        {
            "id": 4,
            "question": "What is the formula for water?",
            "answer": "H2O",
            "difficulty": "medium",
        },
        {
            "id": 5,
            "question": "What is the speed of light?",
            "answer": "299,792,458 m/s",
            "difficulty": "hard",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
        json.dump(data, temp)
        temp_path = temp.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def sample_text_data():
    """Create a temporary text file with sample data."""
    lines = [
        "This is a sample text file.",
        "It contains multiple lines of text.",
        "Each line can be processed separately.",
        "Line processing is an important aspect of NLP.",
        "The final line has some numbers: 123, 456, 789.",
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as temp:
        for line in lines:
            temp.write(line + "\n")
        temp_path = temp.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


def test_json_data_end_to_end(sample_json_data):
    """Test end-to-end JSON data processing pipeline."""

    # Define schema for validation
    schema = {
        "type": "object",
        "required": ["id", "question", "answer"],
        "properties": {
            "id": {"type": "integer"},
            "question": {"type": "string"},
            "answer": {"type": "string"},
            "difficulty": {"type": "string"},
        },
    }

    # Create components
    loader = JsonLoader(file_path=sample_json_data)
    validator = SchemaValidator(schema=schema)

    # Filter for easy questions
    difficulty_filter = Filter(lambda x: x["difficulty"] == "easy")

    # Shuffler for randomization
    shuffler = Shuffler(seed=42)

    # Create data service
    data_service = DataService()
    data_service.set_loader(loader)
    data_service.add_validator(validator)
    data_service.add_transformer(difficulty_filter)
    data_service.add_transformer(shuffler)

    # Load and process the data
    processed_data = data_service.load_and_process()

    # Verify results
    assert len(processed_data) == 3  # Only the "easy" questions
    assert all(item["difficulty"] == "easy" for item in processed_data)

    # Verify the data structure
    for item in processed_data:
        assert "id" in item
        assert "question" in item
        assert "answer" in item
        assert isinstance(item["id"], int)
        assert isinstance(item["question"], str)
        assert isinstance(item["answer"], str)


def test_text_data_end_to_end(sample_text_data):
    """Test end-to-end text data processing pipeline."""

    # Create components
    loader = TextLoader(file_path=sample_text_data)

    # Simple tokenizer that splits on spaces
    tokenizer = Tokenizer(tokenize_fn=lambda x: x.split())

    # Filter lines with numbers
    number_filter = Filter(lambda x: any(char.isdigit() for char in x))

    # Create data service
    data_service = DataService()
    data_service.set_loader(loader)
    data_service.add_transformer(number_filter)
    data_service.add_transformer(tokenizer)

    # Load and process the data
    processed_data = data_service.load_and_process()

    # Verify results
    assert len(processed_data) == 1  # Only the line with numbers
    assert isinstance(processed_data[0], list)  # Should be tokenized

    # The tokenized version of "The final line has some numbers: 123, 456, 789."
    expected_tokens = [
        "The",
        "final",
        "line",
        "has",
        "some",
        "numbers:",
        "123,",
        "456,",
        "789.",
    ]
    assert processed_data[0] == expected_tokens


def test_data_sampling(sample_json_data):
    """Test data sampling within the pipeline."""

    # Create components
    loader = JsonLoader(file_path=sample_json_data)

    # Create sampler for random sampling
    sampler = RandomSampler(sample_size=2, seed=42)

    # Create data service
    data_service = DataService()
    data_service.set_loader(loader)
    data_service.set_sampler(sampler)

    # Load and sample the data
    sampled_data = data_service.load_and_process()

    # Verify results
    assert len(sampled_data) == 2  # Should have 2 items as specified

    # Sample again to verify reproducibility with same seed
    data_service_2 = DataService()
    data_service_2.set_loader(loader)
    data_service_2.set_sampler(RandomSampler(sample_size=2, seed=42))

    sampled_data_2 = data_service_2.load_and_process()

    # Should get the same samples with the same seed
    assert [item["id"] for item in sampled_data] == [
        item["id"] for item in sampled_data_2
    ]

    # Sample with different seed
    data_service_3 = DataService()
    data_service_3.set_loader(loader)
    data_service_3.set_sampler(RandomSampler(sample_size=2, seed=43))

    sampled_data_3 = data_service_3.load_and_process()

    # Might get different samples with different seed
    # This is probabilistic, so we don't assert equality or inequality
