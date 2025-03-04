"""
XCS Tree Utilities

This module provides registration and transformation utilities for immutable
EmberModules and other tree-like objects within the Ember system. It implements
functions analogous to JAX's pytrees:
  • register_tree: Registers a type with its custom flatten and unflatten functions.
  • tree_flatten: Recursively flattens an object into its constituent leaves and
                  auxiliary metadata.
  • tree_unflatten: Reconstructs an object from its auxiliary metadata and flattened leaves.

All functions enforce strong type annotations and require named parameter invocation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, Type, TypeVar, Generic, cast, Protocol, Hashable

# Define more precise type variables for tree operations
T_co = TypeVar('T_co', covariant=True)  # The type being returned (covariant)
T_contra = TypeVar('T_contra', contravariant=True)  # The type being consumed (contravariant)
L = TypeVar('L')  # Leaf value type
A_co = TypeVar('A_co', covariant=True)  # Auxiliary data (covariant when producing)
A_contra = TypeVar('A_contra', contravariant=True)  # Auxiliary data (contravariant when consuming)

# Protocol for flatten function - converts an object to (leaves, auxiliary data)
class FlattenFn(Protocol[T_contra, L, A_co]):
    def __call__(self, obj: T_contra) -> Tuple[List[L], A_co]: ...

# Protocol for unflatten function - reconstructs object from auxiliary data and leaves
class UnflattenFn(Protocol[T_co, L, A_contra]):
    def __call__(self, aux: A_contra, children: List[L]) -> T_co: ...

# Type variable for registry keys
T = TypeVar('T')

# Definition for the Aux type used throughout the module
AuxType = Tuple[Type[object], object]

# Global registry mapping a type to its (flatten, unflatten) functions with improved typing
# We use structural subtyping with Protocol classes instead of Any
_PytreeRegistryType = Dict[
    Type[T],
    Tuple[FlattenFn[T, L, AuxType], UnflattenFn[T, L, AuxType]],
]
_pytree_registry: _PytreeRegistryType = {}


def register_tree(
    *,
    cls: Type[T],
    flatten_func: FlattenFn[T, L, AuxType],
    unflatten_func: UnflattenFn[T, L, AuxType],
) -> None:
    """Registers a type with its custom flatten and unflatten functions for XCS tree utilities.

    Args:
        cls: The type to register.
        flatten_func: A function that flattens an instance of `cls`. It returns a tuple
                      containing a list of leaves and auxiliary metadata.
        unflatten_func: A function that reconstructs an instance of `cls` from auxiliary
                        metadata and a list of leaves.

    Raises:
        ValueError: If `cls` is already registered.
    """
    if cls in _pytree_registry:
        raise ValueError(f"Type {cls.__name__} is already registered as a tree node.")
    _pytree_registry[cls] = (flatten_func, unflatten_func)


def _flatten_iterable(iterable: List[object]) -> Tuple[List[L], List[Tuple[Tuple[Type[object], object], int]]]:
    """Helper function to flatten iterable objects such as lists or tuples.

    Args:
        iterable: The iterable to flatten.

    Returns:
        A tuple containing:
          - A list of flattened leaves.
          - A list of tuples, each containing the child's auxiliary metadata and its leaf count.
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

    Args:
        aux_list: A list of tuples where each tuple contains auxiliary metadata and a leaf count.
        children: The list of flattened leaves.

    Returns:
        A list of unflattened elements.

    Raises:
        ValueError: If the total number of leaves does not match the expected count.
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

    Args:
        aux_list: A list of tuples where each tuple is (key, auxiliary metadata, leaf count).
        children: The list of flattened leaves.

    Returns:
        The reconstructed dictionary.

    Raises:
        ValueError: If the total number of leaves does not match the expected count.
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

    Args:
        tree: The tree-like object to flatten.

    Returns:
        A tuple where the first element is a list of leaves and the second element is the auxiliary
        metadata that encodes the structure of the tree.
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

    Args:
        aux: A tuple containing the tree type and associated auxiliary metadata.
        children: The list of leaves from which to reconstruct the tree.

    Returns:
        The reconstructed tree-like object.

    Raises:
        ValueError: If the provided leaves do not match the expected structure or if an unregistered
                    type with multiple leaves is encountered.
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
        unflattened_seq = _unflatten_sequence(cast(List[Tuple[AuxType, int]], metadata), children)
        return tuple(unflattened_seq)
        
    elif tree_type is dict:
        if not isinstance(metadata, list):
            raise ValueError("Invalid metadata for dict reconstruction.")
        return _unflatten_dict(cast(List[Tuple[Hashable, AuxType, int]], metadata), children)
        
    # Handle leaf nodes
    if len(children) != 1:
        raise ValueError(
            f"Unregistered type {tree_type.__name__} expected a single leaf, got {len(children)}."
        )
    return children[0]
