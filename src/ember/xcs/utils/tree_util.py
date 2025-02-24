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

from typing import Any, Callable, Dict, List, Tuple, Type

# Global registry mapping a type to its (flatten, unflatten) functions.
_PytreeRegistryType = Dict[
    Type[Any],
    Tuple[Callable[[Any], Tuple[List[Any], Any]], Callable[[Any, List[Any]], Any]],
]
_pytree_registry: _PytreeRegistryType = {}


def register_tree(
    *,
    cls: Type[Any],
    flatten_func: Callable[[Any], Tuple[List[Any], Any]],
    unflatten_func: Callable[[Any, List[Any]], Any],
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


def _flatten_iterable(iterable: Any) -> Tuple[List[Any], List[Tuple[Any, int]]]:
    """Helper function to flatten iterable objects such as lists or tuples.

    Args:
        iterable: The iterable to flatten.

    Returns:
        A tuple containing:
          - A list of flattened leaves.
          - A list of tuples, each containing the child's auxiliary metadata and its leaf count.
    """
    flat_leaves: List[Any] = []
    children_info: List[Tuple[Any, int]] = []
    for element in iterable:
        leaves, aux = tree_flatten(tree=element)
        flat_leaves.extend(leaves)
        children_info.append((aux, len(leaves)))
    return flat_leaves, children_info


def _unflatten_sequence(
    aux_list: List[Tuple[Any, int]], children: List[Any]
) -> List[Any]:
    """Helper function to unflatten sequences (lists or tuples) from auxiliary metadata.

    Args:
        aux_list: A list of tuples where each tuple contains auxiliary metadata and a leaf count.
        children: The list of flattened leaves.

    Returns:
        A list of unflattened elements.

    Raises:
        ValueError: If the total number of leaves does not match the expected count.
    """
    result: List[Any] = []
    start: int = 0
    for aux_item, leaf_count in aux_list:
        child_leaves: List[Any] = children[start : start + leaf_count]
        start += leaf_count
        result.append(tree_unflatten(aux=aux_item, children=child_leaves))
    if start != len(children):
        raise ValueError("Mismatch in sequence reconstruction: leftover leaves.")
    return result


def _unflatten_dict(
    aux_list: List[Tuple[Any, Any, int]], children: List[Any]
) -> Dict[Any, Any]:
    """Helper function to unflatten dictionaries from auxiliary metadata.

    Args:
        aux_list: A list of tuples where each tuple is (key, auxiliary metadata, leaf count).
        children: The list of flattened leaves.

    Returns:
        The reconstructed dictionary.

    Raises:
        ValueError: If the total number of leaves does not match the expected count.
    """
    result: Dict[Any, Any] = {}
    start: int = 0
    for key, aux_item, leaf_count in aux_list:
        child_leaves: List[Any] = children[start : start + leaf_count]
        start += leaf_count
        result[key] = tree_unflatten(aux=aux_item, children=child_leaves)
    if start != len(children):
        raise ValueError("Mismatch in dictionary reconstruction: leftover leaves.")
    return result


def tree_flatten(*, tree: Any) -> Tuple[List[Any], Any]:
    """Recursively flattens a tree object into its constituent leaves and auxiliary metadata.

    Args:
        tree: The tree-like object to flatten.

    Returns:
        A tuple where the first element is a list of leaves and the second element is the auxiliary
        metadata that encodes the structure of the tree.
    """
    tree_type: Type[Any] = type(tree)
    if tree_type in _pytree_registry:
        flatten_func, _ = _pytree_registry[tree_type]
        children, aux = flatten_func(tree)
        flat_leaves: List[Any] = []
        for child in children:
            child_leaves, _ = tree_flatten(tree=child)
            flat_leaves.extend(child_leaves)
        return flat_leaves, (tree_type, aux)
    elif isinstance(tree, dict):
        sorted_keys = sorted(tree.keys())
        flat_leaves: List[Any] = []
        children_info: List[Tuple[Any, Any, int]] = []
        for key in sorted_keys:
            leaves, child_aux = tree_flatten(tree=tree[key])
            leaf_count: int = len(leaves)
            children_info.append((key, child_aux, leaf_count))
            flat_leaves.extend(leaves)
        return flat_leaves, (dict, children_info)
    elif isinstance(tree, list):
        flat_leaves, children_info = _flatten_iterable(tree)
        return flat_leaves, (list, children_info)
    elif isinstance(tree, tuple):
        flat_leaves, children_info = _flatten_iterable(tree)
        return flat_leaves, (tuple, children_info)
    else:
        return [tree], (tree_type, None)


def tree_unflatten(*, aux: Any, children: List[Any]) -> Any:
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
    if tree_type in _pytree_registry:
        _, unflatten_func = _pytree_registry[tree_type]
        return unflatten_func(metadata, children)
    if tree_type is list:
        if not isinstance(metadata, list):
            raise ValueError("Invalid metadata for list reconstruction.")
        return _unflatten_sequence(metadata, children)
    elif tree_type is tuple:
        if not isinstance(metadata, list):
            raise ValueError("Invalid metadata for tuple reconstruction.")
        unflattened_seq = _unflatten_sequence(metadata, children)
        return tuple(unflattened_seq)
    elif tree_type is dict:
        if not isinstance(metadata, list):
            raise ValueError("Invalid metadata for dict reconstruction.")
        return _unflatten_dict(metadata, children)
    if len(children) != 1:
        raise ValueError(
            f"Unregistered type {tree_type.__name__} expected a single leaf, got {len(children)}."
        )
    return children[0]
