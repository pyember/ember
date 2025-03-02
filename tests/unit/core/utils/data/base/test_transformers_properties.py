"""
Property-based tests for the ember.core.utils.data.base.transformers module.

This module contains property-based tests using Hypothesis to test invariants
and properties of data transformers.
"""

import pytest
from typing import Dict, Any, List, Optional, Callable
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.extra.pandas import column, data_frames

import pandas as pd
from datasets import Dataset

from ember.core.utils.data.base.transformers import (
    IDatasetTransformer,
    NoOpTransformer,
    DatasetType,
    DatasetItem,
)


# Create strategies for generating test data
@st.composite
def dataset_items(draw, min_fields=1, max_fields=5):
    """Strategy for generating dataset items (dictionaries) with consistent types."""
    num_fields = draw(st.integers(min_value=min_fields, max_value=max_fields))
    
    # Generate field names
    field_names = draw(st.lists(
        st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()),
        min_size=num_fields,
        max_size=num_fields,
        unique=True
    ))
    
    # Generate values for each field - use a consistent type for each field name
    field_values = {}
    
    # First, decide on a type for each field - this ensures consistent types
    # within the same list of dictionaries later
    field_types = {}
    for field in field_names:
        # Limit to simple types that are compatible with Dataset conversion
        field_types[field] = draw(st.sampled_from(['text', 'integer', 'boolean']))
    
    # Now populate values using the chosen types
    for field in field_names:
        value_type = field_types[field]
        
        if value_type == 'text':
            # Use ASCII only to avoid encoding issues
            field_values[field] = draw(st.text(
                alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                min_size=1, 
                max_size=20
            ))
        elif value_type == 'integer':
            field_values[field] = draw(st.integers(min_value=0, max_value=100))
        elif value_type == 'boolean':
            field_values[field] = draw(st.booleans())
    
    return field_values


@st.composite
def dataset_lists(draw, min_items=1, max_items=5):
    """Strategy for generating lists of dataset items with consistent schema."""
    num_items = draw(st.integers(min_value=min_items, max_value=max_items))
    
    # First, choose a common field set to ensure all items have the same schema
    num_fields = draw(st.integers(min_value=1, max_value=3))
    field_names = draw(st.lists(
        st.text(min_size=1, max_size=10).filter(lambda x: x.isalnum()),
        min_size=num_fields,
        max_size=num_fields,
        unique=True
    ))
    
    # Decide on a type for each field - this ensures consistent types
    field_types = {}
    for field in field_names:
        # Limit to simple types that are compatible with Dataset conversion
        field_types[field] = draw(st.sampled_from(['text', 'integer']))
    
    # Generate a list of dataset items
    items = []
    for _ in range(num_items):
        item = {}
        for field in field_names:
            value_type = field_types[field]
            
            if value_type == 'text':
                # Use ASCII only to avoid encoding issues
                item[field] = draw(st.text(
                    alphabet="abcdefghijklmnopqrstuvwxyz",
                    min_size=1, 
                    max_size=5
                ))
            elif value_type == 'integer':
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
            effective_rename = {k: v for k, v in self.rename_map.items() if k in columns}
            
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
        seed=st.integers(min_value=0, max_value=1000)
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
    @given(
        data=dataset_lists(min_items=1, max_items=5)
    )
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
                assert list_result[i][renamed_key] == dataset_result_list[i][renamed_key]
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
        composite = CompositeTransformer([
            NoOpTransformer(),
            NoOpTransformer(),
            NoOpTransformer()
        ])
        
        # Apply the composite
        result = composite.transform(data=data)
        
        # Should be identity
        assert result is data
    
    @given(
        data=dataset_lists(min_items=1, max_items=10),
        num_transformers=st.integers(min_value=1, max_value=5)
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