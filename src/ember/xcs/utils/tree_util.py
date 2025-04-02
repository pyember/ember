"""
XCS Tree Utilities

This module provides registration and transformation utilities for immutable
EmberModules and other tree-like objects within the Ember system. It implements
functions analogous to JAX's pytrees for structural decomposition and reconstruction
of arbitrarily nested data structures:

  • register_tree: Registers a type with its custom flatten and unflatten functions.
  • tree_flatten: Recursively flattens an object into its constituent leaves and
                  auxiliary metadata.
  • tree_unflatten: Reconstructs an object from its auxiliary metadata and flattened leaves.

All functions enforce strong type annotations and require named parameter invocation.

This module enables critical XCS system capabilities:
1. Transforms like vmap and pmap operate on nested structures without manual handling
2. JIT compilation analyzes and optimizes operators by decomposing their structure
3. Execution engines efficiently traverse complex object graphs

Examples:
    # Registering a custom type
    class MyType:
        def __init__(self, value):
            self.value = value

    def flatten_my_type(obj):
        return [obj.value], (type(obj), None)

    def unflatten_my_type(aux_data, children):
        cls, _ = aux_data
        return cls(children[0])

    register_tree(
        cls=MyType,
        flatten_func=flatten_my_type,
        unflatten_func=unflatten_my_type
    )

    # Using tree operations
    obj = {"a": MyType(1), "b": [MyType(2), MyType(3)]}
    leaves, aux_data = tree_flatten(tree=obj)  # leaves = [1, 2, 3]
    reconstructed = tree_unflatten(aux=aux_data, children=leaves)  # Same structure as obj
"""

from __future__ import annotations

import logging
from typing import (
    Dict,
    Hashable,
    List,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    cast,
)

# Define more precise type variables for tree operations
T_co = TypeVar("T_co", covariant=True)  # The type being returned (covariant)
T_contra = TypeVar(
    "T_contra", contravariant=True
)  # The type being consumed (contravariant)
L = TypeVar("L")  # Leaf value type
A_co = TypeVar("A_co", covariant=True)  # Auxiliary data (covariant when producing)
A_contra = TypeVar(
    "A_contra", contravariant=True
)  # Auxiliary data (contravariant when consuming)


# Protocol for flatten function - converts an object to (leaves, auxiliary data)
class FlattenFn(Protocol[T_contra, L, A_co]):
    def __call__(self, obj: T_contra) -> Tuple[List[L], A_co]:
        ...


# Protocol for unflatten function - reconstructs object from auxiliary data and leaves
class UnflattenFn(Protocol[T_co, L, A_contra]):
    def __call__(self, aux: A_contra, children: List[L]) -> T_co:
        ...


# Type variable for registry keys
T = TypeVar("T")

# Definition for the Aux type used throughout the module
AuxType = Tuple[Type[object], object]

# Global registry mapping a type to its (flatten, unflatten) functions with improved typing
# We use structural subtyping with Protocol classes instead of Any
_PytreeRegistryType = Dict[
    Type[T],
    Tuple[FlattenFn[T, L, AuxType], UnflattenFn[T, L, AuxType]],
]
# Initialize the registry
_pytree_registry: _PytreeRegistryType = {}


# Forward import for EmberModel
from ember.core.types.ember_model import EmberModel


# Functions for handling EmberModel in tree operations
def _flatten_ember_model(model: EmberModel) -> Tuple[List[object], AuxType]:
    """Flatten an EmberModel instance into its dictionary and type information.

    Args:
        model: The EmberModel instance to flatten

    Returns:
        A tuple containing:
          - A single-element list with the model's dictionary representation
          - Auxiliary data with the model's class for reconstruction
    """
    # Extract the model's data as a dictionary
    model_dict = model.to_dict()
    
    # Store the exact type information including module path for reliable reconstruction
    model_type = type(model)
    type_path = f"{model_type.__module__}.{model_type.__qualname__}"
    
    return [model_dict], (model_type, type_path)


def _unflatten_ember_model(aux: AuxType, children: List[object]) -> EmberModel:
    """Reconstruct an EmberModel from its dictionary representation.

    Args:
        aux: Auxiliary data containing the model class
        children: List containing the model's dictionary representation

    Returns:
        Reconstructed EmberModel instance
    """
    model_cls, type_path = aux
    model_dict = children[0]

    # Try to import the exact class using the type path
    if isinstance(type_path, str):
        try:
            # Split module and class parts
            last_dot = type_path.rfind('.')
            if last_dot > 0:
                module_name = type_path[:last_dot]
                class_name = type_path[last_dot+1:]
                
                # Import module and get class
                module = __import__(module_name, fromlist=[class_name])
                actual_cls = getattr(module, class_name)
                
                # Reconstruct with proper type
                if hasattr(actual_cls, "from_dict") and callable(getattr(actual_cls, "from_dict")):
                    return actual_cls.from_dict(model_dict)
        except (ImportError, AttributeError) as e:
            # Log and fall back to provided class
            logging.debug(f"Error importing {type_path}: {e}")

    # Fallback to provided class
    if hasattr(model_cls, "from_dict") and callable(getattr(model_cls, "from_dict")):
        return model_cls.from_dict(model_dict)
    
    # Last resort
    return model_cls(**model_dict)


def register_tree(
    *,
    cls: Type[T],
    flatten_func: FlattenFn[T, L, AuxType],
    unflatten_func: UnflattenFn[T, L, AuxType],
) -> None:
    """Registers a type with its custom flatten and unflatten functions for XCS tree utilities.

    Extends the tree traversal system to handle custom types by registering specialized
    functions that control how instances are decomposed and reconstructed. This enables
    transformations like JIT and vmap to work with arbitrary user-defined types.

    The flatten function should extract dynamic values that participate in transformations,
    separating them from static metadata. The unflatten function should reconstruct
    instances from transformed values and the original metadata.

    Args:
        cls: The type to register, which will be used as the key in the registry.
        flatten_func: A function that decomposes an instance of `cls` into a tuple
                      containing a list of dynamic leaf values and static auxiliary metadata.
                      Signature: (obj: T) -> Tuple[List[L], AuxType]
        unflatten_func: A function that reconstructs an instance of `cls` from auxiliary
                        metadata and transformed leaf values.
                        Signature: (aux: AuxType, children: List[L]) -> T

    Raises:
        ValueError: If `cls` is already registered, to prevent conflicts.

    Example:
        ```python
        class CustomNode:
            def __init__(self, value, metadata):
                self.value = value
                self.metadata = metadata  # Static value that won't change

        def flatten_custom_node(node):
            # Return leaves (dynamic values) and auxiliary data (static values)
            return [node.value], (CustomNode, node.metadata)

        def unflatten_custom_node(aux_data, children):
            node_cls, metadata = aux_data
            # Reconstruct with transformed leaves and original metadata
            return node_cls(children[0], metadata)

        register_tree(
            cls=CustomNode,
            flatten_func=flatten_custom_node,
            unflatten_func=unflatten_custom_node
        )
        ```
    """
    if cls in _pytree_registry:
        raise ValueError(f"Type {cls.__name__} is already registered as a tree node.")
    _pytree_registry[cls] = (flatten_func, unflatten_func)


def _flatten_iterable(
    iterable: List[object],
) -> Tuple[List[L], List[Tuple[Tuple[Type[object], object], int]]]:
    """Helper function to flatten iterable objects such as lists or tuples.

    Processes a sequence of elements, flattening each one and consolidating their
    leaves into a single flat list. For each element, it also collects auxiliary
    metadata and keeps track of how many leaves came from that element, creating
    a detailed structural map of the original sequence.

    This implementation recursively flattens each element using tree_flatten,
    allowing it to handle arbitrarily nested structures within sequences.
    The detailed metadata captured enables perfect reconstruction during unflattening.

    Args:
        iterable: The sequence (list or tuple) to flatten, containing arbitrary elements
                 that may themselves be complex nested structures.

    Returns:
        A tuple containing:
          - A list of all flattened leaves extracted from the sequence elements.
          - A list of tuples, each containing:
            * The auxiliary metadata for an element
            * The number of leaves that came from that element

    This metadata structure is crucial for the unflattening process to correctly
    allocate leaves to each element during reconstruction.
    """
    flat_leaves: List[L] = []
    children_info: List[Tuple[Tuple[Type[object], object], int]] = []
    for element in iterable:
        leaves: List[L] = []
        aux: AuxType
        leaves, aux = tree_flatten(tree=element)
        flat_leaves.extend(leaves)
        children_info.append((aux, len(leaves)))
    return flat_leaves, children_info


def _unflatten_sequence(
    aux_list: List[Tuple[AuxType, int]], children: List[L]
) -> List[object]:
    """Helper function to unflatten sequences (lists or tuples) from auxiliary metadata.

    Reconstructs a sequence of elements from flattened leaves and auxiliary data.
    Each entry in aux_list describes one element in the original sequence, including
    metadata about its structure and how many leaves it contains.

    The implementation processes elements sequentially, extracting the appropriate
    number of leaves for each element and recursively reconstructing it using tree_unflatten.
    This ensures that complex nested structures are properly rebuilt.

    Args:
        aux_list: A list of tuples where each tuple contains:
                  - The auxiliary metadata for an element
                  - The number of leaves that element requires
        children: The flat list of leaf values to incorporate into the sequence.

    Returns:
        A list of reconstructed elements that can later be converted to the
        appropriate sequence type (list or tuple).

    Raises:
        ValueError: If the total number of leaves consumed doesn't match the length
                   of the children list, indicating a mismatch between flattening
                   and unflattening operations.
    """
    result: List[object] = []
    start: int = 0
    for aux_item, leaf_count in aux_list:
        child_leaves: List[L] = children[start : start + leaf_count]
        start += leaf_count
        result.append(tree_unflatten(aux=aux_item, children=child_leaves))
    if start != len(children):
        raise ValueError("Mismatch in sequence reconstruction: leftover leaves.")
    return result


# Function for unflattening dictionaries
def _unflatten_dict(
    aux_list: List[Tuple[Hashable, AuxType, int]], children: List[L]
) -> Dict[Hashable, object]:
    """Helper function to unflatten dictionaries from auxiliary metadata.

    Reconstructs a dictionary from flattened leaves and structured auxiliary data.
    The aux_list contains the key, structural metadata, and leaf count for each
    key-value pair in the original dictionary. This function processes each entry
    sequentially, extracting the needed leaves and recursively reconstructing each value.

    Dictionary reconstruction is more complex than sequence reconstruction because
    it needs to preserve both keys and the structure of values. The function maintains
    original key ordering by processing aux_list entries in order.

    Args:
        aux_list: A list of tuples where each tuple contains:
                  - key: The dictionary key (must be hashable)
                  - aux_item: The auxiliary metadata for the value
                  - leaf_count: The number of leaves needed for this value
        children: The flat list of leaf values to incorporate into the dictionary.

    Returns:
        The fully reconstructed dictionary with the original keys and
        reconstructed values.

    Raises:
        ValueError: If the total number of leaves consumed doesn't match the length
                   of the children list, indicating an inconsistency between the
                   flattening and unflattening operations.
    """
    result: Dict[Hashable, object] = {}
    start: int = 0
    for key, aux_item, leaf_count in aux_list:
        child_leaves: List[L] = children[start : start + leaf_count]
        start += leaf_count
        result[key] = tree_unflatten(aux=aux_item, children=child_leaves)
    if start != len(children):
        raise ValueError("Mismatch in dictionary reconstruction: leftover leaves.")
    return result


def tree_flatten(*, tree: object) -> Tuple[List[L], AuxType]:
    """Recursively flattens a tree object into its constituent leaves and auxiliary metadata.

    Traverses a nested data structure and extracts all leaf values into a flat list while
    preserving structural information in auxiliary metadata. This decomposition allows
    transformations to operate on leaf values while maintaining the ability to reconstruct
    the original structure.

    The function handles four cases:
    1. Registered custom types - Uses their custom flatten_func
    2. Dictionaries - Flattens values while preserving keys in auxiliary data
    3. Lists/tuples - Flattens elements while preserving sequence type
    4. Other values - Treated as leaf nodes

    Args:
        tree: The tree-like object to flatten, which may be a nested structure
              containing dictionaries, lists, tuples, and custom registered types.

    Returns:
        A tuple containing:
          - A flat list of all leaf values extracted from the tree
          - Auxiliary metadata that encodes the structure information needed for reconstruction

    Example:
        ```python
        # For a nested structure
        data = {"a": [1, 2], "b": {"c": 3}}

        # Flatten into leaves and auxiliary data
        leaves, aux_data = tree_flatten(tree=data)

        # leaves = [1, 2, 3]
        # aux_data contains nested structure information

        # Can be reconstructed exactly with tree_unflatten
        reconstructed = tree_unflatten(aux=aux_data, children=leaves)
        # reconstructed = {"a": [1, 2], "b": {"c": 3}}
        ```
    """
    tree_type: Type[object] = type(tree)

    # Handle registered types via their custom flatten function
    if tree_type in _pytree_registry:
        flatten_func, _ = _pytree_registry[tree_type]
        children, aux = flatten_func(tree)
        flat_leaves: List[L] = []
        for child in children:
            child_leaves: List[L] = []
            child_aux: AuxType
            child_leaves, child_aux = tree_flatten(tree=child)
            flat_leaves.extend(child_leaves)
        return flat_leaves, (tree_type, aux)

    # Handle dictionaries specially
    elif isinstance(tree, dict):
        sorted_keys = sorted(tree.keys())
        dict_leaves: List[L] = []
        # For dictionaries, the auxiliary data has three components: key, aux data, and leaf count
        dict_children_info: List[Tuple[Hashable, AuxType, int]] = []

        for key in sorted_keys:
            item_dict = cast(Dict[Hashable, object], tree)
            dict_item_leaves: List[L] = []
            dict_item_aux: AuxType
            dict_item_leaves, dict_item_aux = tree_flatten(tree=item_dict[key])
            leaf_count: int = len(dict_item_leaves)
            dict_children_info.append((key, dict_item_aux, leaf_count))
            dict_leaves.extend(dict_item_leaves)

        return dict_leaves, (dict, dict_children_info)

    # Handle lists and tuples with a common helper
    elif isinstance(tree, (list, tuple)):
        # For lists and tuples, process with the common iterable flattener
        flat_leaves, children_info = _flatten_iterable(cast(List[object], tree))
        return flat_leaves, (tree_type, children_info)

    # Base case: a leaf node
    else:
        return [cast(L, tree)], (tree_type, None)


def tree_unflatten(*, aux: AuxType, children: List[L]) -> object:
    """Reconstructs an object from its auxiliary metadata and a list of leaves.

    This function is the inverse of tree_flatten, rebuilding a nested structure from
    a flat list of leaf values and the structural metadata. It reconstructs the
    exact structure using the type information and auxiliary data stored during flattening.

    The reconstruction process follows these steps:
    1. Extract type and metadata from the auxiliary data
    2. If the type is registered, use its custom unflatten function
    3. For built-in containers (list, tuple, dict), reconstruct using helper functions
    4. For leaf nodes, return the single child value

    This operation is crucial for transformations that need to maintain structural
    integrity while operating on the underlying values.

    Args:
        aux: A tuple containing (type_info, metadata) where:
             - type_info: The Python type of the original object
             - metadata: Structure-specific auxiliary data needed for reconstruction
        children: The flat list of leaf values to incorporate into the reconstructed structure.
                 These should match the structure expected based on the auxiliary data.

    Returns:
        The reconstructed tree-like object with the original structure but potentially
        transformed leaf values.

    Raises:
        ValueError: If the provided leaves don't match the expected structure (e.g., incorrect
                   number of leaves) or if an unregistered type with multiple leaves is encountered.
                   This helps catch inconsistencies between flattening and unflattening operations.

    Example:
        ```python
        # Starting with flattened data
        leaves = [1, 2, 3]
        aux_data = (dict, [...])  # Auxiliary data encoding a dictionary structure

        # Reconstruct the original structure
        result = tree_unflatten(aux=aux_data, children=leaves)
        # result might be something like {"a": [1, 2], "b": 3}
        ```
    """
    tree_type, metadata = aux

    # Handle registered types via their custom unflatten function
    if tree_type in _pytree_registry:
        _, unflatten_func = _pytree_registry[tree_type]
        # Cast the metadata to the expected type for the unflatten function
        typed_metadata = cast(AuxType, metadata)
        return unflatten_func(typed_metadata, children)

    # Handle built-in container types
    if tree_type is list:
        if not isinstance(metadata, list):
            raise ValueError("Invalid metadata for list reconstruction.")
        return _unflatten_sequence(cast(List[Tuple[AuxType, int]], metadata), children)

    elif tree_type is tuple:
        if not isinstance(metadata, list):
            raise ValueError("Invalid metadata for tuple reconstruction.")
        unflattened_seq = _unflatten_sequence(
            cast(List[Tuple[AuxType, int]], metadata), children
        )
        return tuple(unflattened_seq)

    elif tree_type is dict:
        if not isinstance(metadata, list):
            raise ValueError("Invalid metadata for dict reconstruction.")
        return _unflatten_dict(
            cast(List[Tuple[Hashable, AuxType, int]], metadata), children
        )

    # Handle leaf nodes
    if len(children) != 1:
        raise ValueError(
            f"Unregistered type {tree_type.__name__} expected a single leaf, got {len(children)}."
        )
    return children[0]


# Register EmberModel with the tree utilities system after all functions are defined
register_tree(
    cls=EmberModel,
    flatten_func=_flatten_ember_model,
    unflatten_func=_unflatten_ember_model,
)
