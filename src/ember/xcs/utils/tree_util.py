"""
XCS Tree Utilities

This module provides our own registration and transformation utilities
for immutable EmberModules (and other tree-like objects) in the Ember system.
It implements functions analogous to JAX's pytrees:
  • register_tree: Register a type with its custom flatten/unflatten functions.
  • tree_flatten: Recursively flatten an object into a list of leaves and auxiliary metadata.
  • tree_unflatten: Reconstruct an object from its auxiliary metadata and flattened leaves.

All functions use strong type annotations and named parameter invocation.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Tuple, Type, Optional
from dataclasses import is_dataclass

# Global registry mapping a type to its (flatten, unflatten) functions.
_pytree_registry: Dict[
    Type[Any],
    Tuple[Callable[[Any], Tuple[List[Any], Any]], Callable[[Any, List[Any]], Any]],
] = {}


def register_tree(
    *,
    cls: Type[Any],
    flatten_func: Callable[[Any], Tuple[List[Any], Any]],
    unflatten_func: Callable[[Any, List[Any]], Any],
) -> None:
    """
    Registers a type with its flatten/unflatten functions for XCS tree utilities.
    If the type is already registered, raises a ValueError.
    """
    if cls in _pytree_registry:
        raise ValueError(f"Type {cls.__name__} is already registered as a tree node.")
    _pytree_registry[cls] = (flatten_func, unflatten_func)


def tree_flatten(*, tree: Any) -> Tuple[List[Any], Any]:
    """
    Recursively flattens a tree object into a list of leaves (flat_leaves)
    and auxiliary metadata (aux).

    If the type of `tree` is registered, its flatten function is used. Otherwise,
    built-in containers (list, tuple, dict) are handled generically, and unregistered
    objects are treated as leaves.
    """
    tree_type = type(tree)
    # Check if we have a custom flatten function for this type
    if tree_type in _pytree_registry:
        flatten_func, _ = _pytree_registry[tree_type]
        children, aux = flatten_func(tree)
        flat_leaves: List[Any] = []
        for child in children:
            child_leaves, _ = tree_flatten(tree=child)
            flat_leaves.extend(child_leaves)
        # We'll store (tree_type, aux) as the overall 'aux'
        return flat_leaves, (tree_type, aux)
    elif isinstance(tree, (list, tuple)):
        # Flatten each element
        flat_leaves: List[Any] = []
        aux = (tree_type, len(tree))
        for item in tree:
            leaves, _ = tree_flatten(tree=item)
            flat_leaves.extend(leaves)
        return flat_leaves, aux
    elif isinstance(tree, dict):
        # Flatten each value in sorted key order for determinism
        keys = sorted(tree.keys())
        flat_leaves: List[Any] = []
        for key in keys:
            leaves, _ = tree_flatten(tree=tree[key])
            flat_leaves.extend(leaves)
        return flat_leaves, (dict, keys)
    else:
        # Treat non-registered, non-built-in-collection objects as a single leaf
        return [tree], (tree_type, None)


def tree_unflatten(*, aux: Any, children: List[Any]) -> Any:
    """
    Reconstructs (unflattens) an object from the supplied metadata (aux) and list of leaves (children).

    The 'aux' is expected to be a tuple (tree_type, metadata). If the type is in the registry,
    we call the custom unflatten function. Otherwise, we handle built-in containers or
    unregistered leaf objects.
    """
    tree_type, metadata = aux

    # If we have a custom unflatten function for tree_type, use it
    if tree_type in _pytree_registry:
        _, unflatten_func = _pytree_registry[tree_type]
        return unflatten_func(metadata, children)

    # Otherwise handle built-in types
    if tree_type is list:
        return [children[i] for i in range(metadata)]
    elif tree_type is tuple:
        return tuple(children[i] for i in range(metadata))
    elif tree_type is dict:
        # Reconstruct in the same sorted key order
        result = {}
        # We'll need to pull sub-leaf slices from 'children'
        # but since we don't have a direct mapping of how many leaves belong to each key,
        # a simple approach is to flatten each child again if needed. A more robust approach
        # would store the child's size in 'metadata'. For now, we assume each child is a single leaf.
        keys = metadata
        if len(keys) != len(children):
            raise ValueError("Mismatch in dictionary reconstruction: keys vs leaves.")
        for i, k in enumerate(keys):
            result[k] = children[i]
        return result

    # Unregistered type with multiple children is ambiguous
    if len(children) != 1:
        raise ValueError(
            f"Unregistered type {tree_type.__name__} expected a single leaf, got {len(children)}"
        )
    return children[0]
