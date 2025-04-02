"""
Property-based tests for the ember.core.utils.data.base.transformers module.

This module contains property-based tests using Hypothesis to test invariants
and properties of data transformers.
"""

from typing import Any, Dict, List, Optional, Callable

import pandas as pd
import pytest
from datasets import Dataset
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ember.api.data import DatasetBuilder
from ember.core.utils.data.base.transformers import (
    DatasetType,
    IDatasetTransformer,
    NoOpTransformer,
)


# Create strategies for generating test data
@st.composite
def dataset_items(draw, min_fields=1, max_fields=5):
    """Strategy for generating dataset items (dictionaries) with consistent types."""
    num_fields = draw(st.integers(min_value=min_fields, max_value=max_fields))

    # Generate field names
    field_names = draw(
        st.lists(
            st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()),
            min_size=num_fields,
            max_size=num_fields,
            unique=True,
        )
    )

    # Generate values for each field - use a consistent type for each field name
    field_values = {}

    # First, decide on a type for each field - this ensures consistent types
    # within the same list of dictionaries later
    field_types = {}
    for field in field_names:
        # Limit to simple types that are compatible with Dataset conversion
        field_types[field] = draw(st.sampled_from(["text", "integer", "boolean"]))

    # Now populate values using the chosen types
    for field in field_names:
        value_type = field_types[field]

        if value_type == "text":
            # Use ASCII only to avoid encoding issues
            field_values[field] = draw(
                st.text(
                    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    min_size=1,
                    max_size=20,
                )
            )
        elif value_type == "integer":
            field_values[field] = draw(st.integers(min_value=0, max_value=100))
        elif value_type == "boolean":
            field_values[field] = draw(st.booleans())

    return field_values


@st.composite
def dataset_lists(draw, min_items=1, max_items=5):
    """Strategy for generating lists of dataset items with consistent schema."""
    num_items = draw(st.integers(min_value=min_items, max_value=max_items))

    # First, choose a common field set to ensure all items have the same schema
    num_fields = draw(st.integers(min_value=1, max_value=3))
    field_names = draw(
        st.lists(
            st.text(min_size=1, max_size=10).filter(lambda x: x.isalnum()),
            min_size=num_fields,
            max_size=num_fields,
            unique=True,
        )
    )

    # Decide on a type for each field - this ensures consistent types
    field_types = {}
    for field in field_names:
        # Limit to simple types that are compatible with Dataset conversion
        field_types[field] = draw(st.sampled_from(["text", "integer"]))

    # Generate a list of dataset items
    items = []
    for _ in range(num_items):
        item = {}
        for field in field_names:
            value_type = field_types[field]

            if value_type == "text":
                # Use ASCII only to avoid encoding issues
                item[field] = draw(
                    st.text(
                        alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=5
                    )
                )
            elif value_type == "integer":
                item[field] = draw(st.integers(min_value=0, max_value=50))
        items.append(item)

    return items


# Helper function to convert list to Dataset
def list_to_dataset(data_list: List[Dict[str, Any]]) -> Dataset:
    """Convert a list of dictionaries to a HuggingFace Dataset with robust type handling."""
    if not data_list:
        return Dataset.from_dict({})

    # First, standardize the schema so all items have the same keys
    all_keys = set()
    for item in data_list:
        all_keys.update(item.keys())

    # Create normalized data with consistent types
    normalized = []
    for item in data_list:
        normalized_item = {}
        for key in all_keys:
            # Provide default values for missing keys
            if key not in item:
                # Use empty string for missing text fields
                normalized_item[key] = ""
            else:
                normalized_item[key] = item[key]
        normalized.append(normalized_item)

    # Convert to pandas DataFrame with explicit dtypes for each column
    df = pd.DataFrame(normalized)

    # Then convert to Dataset
    return Dataset.from_pandas(df)


class TestNoOpTransformerProperties:
    """Property-based tests for the NoOpTransformer class."""

    @given(data=dataset_lists())
    def test_noop_identity_property_list(self, data):
        """Property: NoOpTransformer should always return the identical list."""
        transformer = NoOpTransformer()
        result = transformer.transform(data=data)

        # For list inputs, we should get the same list back (identity)
        assert result is data

    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    @given(data=dataset_lists(min_items=1, max_items=5))
    def test_noop_identity_property_dataset(self, data):
        """Property: NoOpTransformer should preserve Dataset contents."""
        # Convert list to Dataset
        dataset = list_to_dataset(data)

        transformer = NoOpTransformer()
        result = transformer.transform(data=dataset)

        # For Dataset inputs, contents should be preserved
        assert isinstance(result, Dataset)
        assert len(result) == len(dataset)

        # Convert back to lists for easier comparison
        result_list = [item for item in result]
        dataset_list = [item for item in dataset]

        assert result_list == dataset_list


class CustomKeyRenameTransformer(IDatasetTransformer):
    """Transformer that renames keys in the dataset."""

    def __init__(self, rename_map: Dict[str, str]):
        """Initialize with a mapping of old_key -> new_key."""
        self.rename_map = rename_map

    def transform(self, *, data: DatasetType) -> DatasetType:
        """Transform the dataset by renaming keys according to the rename map."""
        if isinstance(data, Dataset):
            # For Dataset objects, we need to be careful about column renaming
            # Get current column set
            columns = data.column_names

            # Create rename map limited to columns that exist
            effective_rename = {
                k: v for k, v in self.rename_map.items() if k in columns
            }

            # Get a pandas DataFrame and rename columns
            df = data.to_pandas()
            df = df.rename(columns=effective_rename)

            # Convert back to Dataset
            return Dataset.from_pandas(df)
        else:
            # For list of dicts, transform each item
            result = []
            for item in data:
                new_item = {}
                for key, value in item.items():
                    new_key = self.rename_map.get(key, key)
                    new_item[new_key] = value
                result.append(new_item)
            return result


class TestCustomTransformerProperties:
    """Property-based tests for custom transformers."""

    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    @given(
        data=dataset_lists(min_items=1, max_items=5),
        seed=st.integers(min_value=0, max_value=1000),
    )
    def test_key_rename_property(self, data, seed):
        """Property: renaming and then renaming back should be equivalent to no-op."""
        # Skip if data is empty
        if not data or not data[0]:
            return

        # Take a key from the first item to use in our rename test
        first_key = list(data[0].keys())[0]
        renamed_key = f"{first_key}_renamed_{seed}"

        # Create forward and reverse rename maps
        rename_map = {first_key: renamed_key}
        reverse_map = {renamed_key: first_key}

        # Apply transformations
        forward_transformer = CustomKeyRenameTransformer(rename_map)
        reverse_transformer = CustomKeyRenameTransformer(reverse_map)

        # Apply both transformations
        intermediate = forward_transformer.transform(data=data)
        result = reverse_transformer.transform(data=intermediate)

        # Result should be equivalent to original except for the renamed key
        assert len(result) == len(data)
        for i, original_item in enumerate(data):
            result_item = result[i]
            assert len(result_item) == len(original_item)

            # All values should match
            for key, value in original_item.items():
                assert result_item[key] == value

    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    @given(data=dataset_lists(min_items=1, max_items=5))
    def test_dataset_list_equivalence(self, data):
        """Property: transforming a list or its Dataset equivalent should yield equivalent results."""
        # Skip if empty data
        if not data:
            return

        # Find a key to rename if data exists
        try:
            first_key = list(data[0].keys())[0]
            renamed_key = f"{first_key}_renamed"

            # Create rename map
            rename_map = {first_key: renamed_key}
            transformer = CustomKeyRenameTransformer(rename_map)

            # Transform the list
            list_result = transformer.transform(data=data)

            # Transform the Dataset
            dataset = list_to_dataset(data)
            dataset_result = transformer.transform(data=dataset)

            # Convert Dataset result back to list for comparison
            dataset_result_list = [item for item in dataset_result]

            # Compare transformations by checking renamed key is present
            assert len(list_result) == len(dataset_result_list)
            for i in range(len(list_result)):
                # Check that the renamed key exists in both results
                assert renamed_key in list_result[i]
                assert renamed_key in dataset_result_list[i]
                # Check that the renamed key has the same value
                assert (
                    list_result[i][renamed_key] == dataset_result_list[i][renamed_key]
                )
        except (IndexError, KeyError):
            # Skip test if data structure doesn't support the operations
            pytest.skip("Data structure doesn't have required keys")


class CompositeTransformer(IDatasetTransformer):
    """Transformer that applies multiple transforms in sequence."""

    def __init__(self, transformers: List[IDatasetTransformer]):
        """Initialize with a list of transformers to apply in sequence."""
        self.transformers = transformers

    def transform(self, *, data: DatasetType) -> DatasetType:
        """Apply all transformers in sequence."""
        result = data
        for transformer in self.transformers:
            result = transformer.transform(data=result)
        return result


class TestCompositeTransformerProperties:
    """Property-based tests for composite transformers."""

    @given(data=dataset_lists())
    def test_composite_with_noops(self, data):
        """Property: A composite of NoOp transformers should be equivalent to a single NoOp."""
        # Create a composite of multiple NoOp transformers
        composite = CompositeTransformer(
            [NoOpTransformer(), NoOpTransformer(), NoOpTransformer()]
        )

        # Apply the composite
        result = composite.transform(data=data)

        # Should be identity
        assert result is data

    @given(
        data=dataset_lists(min_items=1, max_items=10),
        num_transformers=st.integers(min_value=1, max_value=5),
    )
    def test_composite_associativity(self, data, num_transformers):
        """Property: Different groupings of transformers should yield the same result."""
        # Skip if empty data
        if not data or not data[0]:
            return

        try:
            # Create a list of key rename transformers
            transformers = []
            keys = list(data[0].keys())
            if not keys:
                return

            # Create transformers that swap a key with a temporary name and back
            for i in range(min(num_transformers, len(keys))):
                key = keys[i]
                temp_key = f"{key}_temp"

                # Add transformer that renames key -> temp_key
                transformers.append(CustomKeyRenameTransformer({key: temp_key}))

                # Add transformer that renames temp_key -> key
                transformers.append(CustomKeyRenameTransformer({temp_key: key}))

            # Create different groupings:
            # 1. Apply all transformers in sequence
            composite1 = CompositeTransformer(transformers)
            result1 = composite1.transform(data=data)

            # 2. Apply transformers in two groups
            midpoint = len(transformers) // 2
            group1 = CompositeTransformer(transformers[:midpoint])
            group2 = CompositeTransformer(transformers[midpoint:])

            intermediate = group1.transform(data=data)
            result2 = group2.transform(data=intermediate)

            # Results should be equivalent
            assert len(result1) == len(result2)
            for i in range(len(result1)):
                assert result1[i] == result2[i]

        except (IndexError, KeyError):
            # Skip test if data structure doesn't support the operations
            pytest.skip("Data structure doesn't have required keys")


class TestDatasetBuilderTransformers:
    """Tests for transformers created through the DatasetBuilder."""

    @given(data=dataset_lists(min_items=1, max_items=5))
    def test_function_transformer_adapter(self, data):
        """Test that function transformer adapters created by DatasetBuilder work correctly."""
        # Skip if data is empty
        if not data:
            return

        # Create a transformation function
        def add_prefix_to_fields(item: Dict[str, Any]) -> Dict[str, Any]:
            return {f"prefix_{k}": v for k, v in item.items()}

        # Create a builder and add the transformer
        builder = DatasetBuilder()
        builder.transform(add_prefix_to_fields)

        # Extract and apply the transformer
        transformer = builder._transformers[0]
        result = transformer.transform(data=data)

        # Verify transformation was applied correctly
        assert len(result) == len(data)
        for i, item in enumerate(result):
            # Check that all original keys are prefixed
            for key, value in data[i].items():
                prefixed_key = f"prefix_{key}"
                assert prefixed_key in item
                assert item[prefixed_key] == value

    @given(
        data=dataset_lists(min_items=1, max_items=5),
        seed1=st.integers(min_value=1, max_value=100),
        seed2=st.integers(min_value=101, max_value=200),
    )
    def test_multiple_function_transformers(self, data, seed1, seed2):
        """Test that multiple function transformers can be chained in a DatasetBuilder."""
        # Skip if data is empty
        if not data:
            return

        # Create two transformation functions
        def add_field_1(item: Dict[str, Any]) -> Dict[str, Any]:
            result = item.copy()
            result[f"added_field_{seed1}"] = seed1
            return result

        def add_field_2(item: Dict[str, Any]) -> Dict[str, Any]:
            result = item.copy()
            result[f"added_field_{seed2}"] = seed2
            return result

        # Create a builder and add both transformers
        builder = DatasetBuilder()
        builder.transform(add_field_1)
        builder.transform(add_field_2)

        # Apply transformations in sequence
        first_transformer = builder._transformers[0]
        second_transformer = builder._transformers[1]

        intermediate = first_transformer.transform(data=data)
        result = second_transformer.transform(data=intermediate)

        # Verify transformation was applied correctly
        assert len(result) == len(data)
        for i, item in enumerate(result):
            # Check that all original data is preserved
            for key, value in data[i].items():
                assert item[key] == value

            # Check that both fields were added
            assert item[f"added_field_{seed1}"] == seed1
            assert item[f"added_field_{seed2}"] == seed2


class TestTransformerContextPreservation:
    """Tests that transformers correctly preserve context between elements."""

    @given(data=dataset_lists(min_items=2, max_items=10))
    def test_stateful_transformer(self, data):
        """Test that stateful transformers can maintain state across elements."""
        # Skip if data is empty
        if not data:
            return

        # Create a stateful transformer that indexes items
        class IndexingTransformer(IDatasetTransformer):
            def transform(self, *, data: DatasetType) -> DatasetType:
                if isinstance(data, Dataset):
                    # For Dataset, convert to list, transform, and convert back
                    items = [item for item in data]
                    transformed = self._index_items(items)
                    return list_to_dataset(transformed)
                else:
                    # For list, directly transform
                    return self._index_items(data)
            
            def _index_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                result = []
                # Keep track of running sums for integer fields encountered so far
                current_sums = {}
                for i, item in enumerate(items):
                    new_item = item.copy()
                    new_item["index"] = i
                    item_sums = {}
                    for key, value in item.items():
                        if isinstance(value, int):
                            # Update the overall running sum for this key
                            current_sums[key] = current_sums.get(key, 0) + value
                            # Store the running sum *up to this item* in the item itself
                            item_sums[f"sum_{key}"] = current_sums[key]
                    new_item.update(item_sums)
                    result.append(new_item)
                return result

        # Apply the transformer
        transformer = IndexingTransformer()
        result = transformer.transform(data=data)

        # Verify transformation was applied correctly
        assert len(result) == len(data)
        
        # Check indices
        for i, item in enumerate(result):
            assert item["index"] == i
            
        # Check running sums for integer fields (starting from second item)
        for i in range(1, len(result)):
            for key, value in data[i].items():
                if isinstance(value, int):
                    running_sum_key = f"sum_{key}"
                    if running_sum_key in result[i]:
                        # Calculate expected sum
                        expected_sum = 0
                        for j in range(i + 1):
                            if key in data[j] and isinstance(data[j][key], int):
                                expected_sum += data[j][key]
                        assert result[i][running_sum_key] == expected_sum


class TestDataPreservationProperties:
    """Tests for data preservation in transformers."""

    @given(data=dataset_lists(min_items=1, max_items=5))
    def test_field_addition_preserves_data(self, data):
        """Test that adding fields preserves all original data."""
        # Skip if data is empty
        if not data:
            return

        # Create a field-adding transformer
        def add_fields(item: Dict[str, Any]) -> Dict[str, Any]:
            result = item.copy()
            result["new_field_1"] = "value1"
            result["new_field_2"] = 42
            return result

        # Create a builder and add the transformer
        builder = DatasetBuilder()
        builder.transform(add_fields)
        transformer = builder._transformers[0]

        # Apply the transformer
        result = transformer.transform(data=data)

        # Verify original data is preserved
        assert len(result) == len(data)
        for i, original_item in enumerate(result):
            for key, value in data[i].items():
                assert result[i][key] == value

    @settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
    @given(
        data=dataset_lists(min_items=1, max_items=5),
        field_name=st.text(min_size=1, max_size=10).filter(lambda x: x.isalnum())
    )
    def test_conditional_transformation(self, data, field_name):
        """Test that transformers can conditionally modify items."""
        # Skip if data is empty or field_name conflicts
        if not data:
            return
            
        # Create a unique field name that won't conflict
        field_name = f"special_{field_name}"
        
        # Skip if field already exists in data
        for item in data:
            if field_name in item:
                return

        # Create a conditional transformer that only modifies items meeting a condition
        def conditional_transform(item: Dict[str, Any]) -> Dict[str, Any]:
            result = item.copy()
            
            # Find numeric fields and conditionally tag items - lower threshold to
            # ensure we have enough matches
            has_numeric = False
            for key, value in item.items():
                if isinstance(value, int) and value > 10:  # lower threshold
                    has_numeric = True
                    break
                    
            if has_numeric:
                result[field_name] = True
                
            return result

        # Create a builder and add the transformer
        builder = DatasetBuilder()
        builder.transform(conditional_transform)
        transformer = builder._transformers[0]

        # Apply the transformer
        result = transformer.transform(data=data)

        # Verify transformation was applied correctly
        assert len(result) == len(data)
        for i, item in enumerate(result):
            # Check that all original data is preserved
            for key, value in data[i].items():
                assert item[key] == value
                
            # Verify the condition was applied correctly
            has_numeric = any(
                isinstance(value, int) and value > 10  # match the lowered threshold
                for key, value in data[i].items()
            )
            
            if has_numeric:
                assert field_name in item
                assert item[field_name] is True
            else:
                assert field_name not in item