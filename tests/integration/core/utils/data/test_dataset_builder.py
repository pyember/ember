"""Integration tests for the DatasetBuilder API."""

import pytest

from ember.api.data import (
    DatasetBuilder,
    Dataset,
    DatasetEntry,
    list_available_datasets,
)


def test_dataset_builder_basic_functionality():
    """Test the basic functionality of the DatasetBuilder API.

    This test just creates a builder and verifies its interface is correct.
    """
    # Create a builder
    builder = DatasetBuilder()

    # Verify it has the expected methods
    assert hasattr(builder, "from_registry")
    assert hasattr(builder, "subset")
    assert hasattr(builder, "split")
    assert hasattr(builder, "sample")
    assert hasattr(builder, "seed")
    assert hasattr(builder, "transform")
    assert hasattr(builder, "config")
    assert hasattr(builder, "build")

    # Verify methods return self for chaining
    assert builder.split("test") is builder
    assert builder.sample(100) is builder
    assert builder.seed(42) is builder
    assert builder.config(param="value") is builder


def test_dataset_builder_method_setting():
    """Test that the DatasetBuilder methods set the right attributes."""
    builder = DatasetBuilder()

    # Call methods
    builder.from_registry("mmlu")
    builder.subset("high_school_mathematics")
    builder.split("test")
    builder.sample(10)
    builder.seed(42)
    builder.config(param1="value1", param2="value2")

    # Check builder state
    assert builder._dataset_name == "mmlu"
    assert builder._config["subset"] == "high_school_mathematics"
    assert builder._split == "test"
    assert builder._sample_size == 10
    assert builder._seed == 42
    assert builder._config["param1"] == "value1"
    assert builder._config["param2"] == "value2"


def test_transform_method_adds_transformers():
    """Test that the transform method adds transformers to the list."""
    builder = DatasetBuilder()

    # Define a simple transformer function
    def transformer_fn(item):
        item["transformed"] = True
        return item

    # Add the transformer
    builder.transform(transformer_fn)

    # Check that we have one transformer in the list
    assert len(builder._transformers) == 1


def test_list_available_datasets():
    """Test the list_available_datasets helper function."""
    # Get the available datasets
    datasets = list_available_datasets()

    # Verify we get a list back
    assert isinstance(datasets, list)

    # There should be at least some default datasets
    assert len(datasets) > 0

    # Some known datasets should be in the list
    for ds in ["mmlu", "truthful_qa"]:
        if ds in datasets:
            return  # At least one known dataset is found

    pytest.fail("No known datasets found in the registry")
