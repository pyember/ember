"""
Compact Network of Networks (NON) Graph Notation

A concise, efficient notation for defining NON architectures with minimal syntax
overhead. Enables rapid prototyping and composition of sophisticated network
architectures through a dense, expressive format.

Format:
- Basic operator: "count:type:model:temperature"
  - count: Number of units/instances
  - type: Operator type code (E=Ensemble, J=Judge, etc.)
  - model: Model identifier
  - temperature: Temperature setting (0.0-1.0)

- Graph: List of operators to be executed sequentially
  [op1, op2, op3]

- Reference operators: "$name"
  - Reference a previously defined operator by name

Examples:
  # Basic pipeline: Ensemble (3 GPT-4o instances) + Judge (Claude 3.5, temp=0)
  ["3:E:gpt-4o:0.7", "1:J:claude-3-5-sonnet:0.0"]

  # Nested structures
  [["3:E:gpt-4o:0.7", "1:V:gpt-4o:0.0"],
   ["3:E:claude-3-5-haiku:1.0", "1:V:claude-3-5-haiku:0.0"],
   "1:J:claude-3-5-sonnet:0.0"]

  # Named components for reuse
  ops = {"sub1": ["3:E:gpt-4o:0.7", "1:V:gpt-4o:0.0"],
         "sub2": ["3:E:claude-3-5:0.7", "1:V:claude-3-5:0.0"]}
  ["$sub1", "$sub2", "1:J:claude-3-5-sonnet:0.0"]
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Union

from ember.core.non import (
    JudgeSynthesis,
    MostCommon,
    Operator,
    Sequential,
    UniformEnsemble,
    Verifier,
    PromptEnhancer,
)

# Type aliases
NodeSpec = Union[str, List[Any], Operator]
GraphSpec = Union[NodeSpec, List[NodeSpec]]
RefMap = Dict[str, Any]

# Type for operator factory functions
OpFactory = Callable[[int, str, float], Operator]  # count, model, temp -> op


class OpRegistry:
    """Registry of operator type codes to factory functions.

    Maps compact notation codes (like "E" for Ensemble) to factory functions
    that create the corresponding operators. Follows instance-based design
    to support isolated configuration contexts and enable clean dependency
    injection.

    This registry is separate from the component reference map used in
    build_graph, maintaining clear separation of concerns between operator
    types and component instances.
    """

    def __init__(self) -> None:
        """Initialize a new registry with an empty factories dictionary."""
        self._factories: Dict[str, OpFactory] = {}

    def register(self, code: str, factory: OpFactory) -> None:
        """Register an operator factory by type code.

        Args:
            code: The operator type code (e.g., "E", "J", "V")
            factory: Function that takes (count, model, temp) and returns an Operator
        """
        self._factories[code] = factory

    def create(self, code: str, count: int, model: str, temp: float) -> Operator:
        """Create an operator instance from a type code and parameters.

        Args:
            code: The operator type code
            count: Number of units/instances
            model: Model identifier
            temp: Temperature setting

        Returns:
            An instantiated operator

        Raises:
            ValueError: If the type code is not registered
        """
        if code not in self._factories:
            codes = ", ".join(sorted(self._factories.keys()))
            raise ValueError(
                f"Unknown operator type code: '{code}'. Valid codes: {codes}"
            )
        return self._factories[code](count, model, temp)

    def has_type(self, code: str) -> bool:
        """Check if a type code is registered.

        Args:
            code: Operator type code to check

        Returns:
            True if the code is registered, False otherwise
        """
        return code in self._factories

    def get_codes(self) -> List[str]:
        """Get a list of all registered type codes.

        Returns:
            A sorted list of registered type codes
        """
        return sorted(self._factories.keys())

    @classmethod
    def create_standard_registry(cls) -> "OpRegistry":
        """Create a new registry with standard operator types pre-registered.

        Returns:
            OpRegistry configured with all standard operators
        """
        registry = cls()

        # Ensemble operators
        registry.register(
            "E",
            lambda c, m, t: UniformEnsemble(num_units=c, model_name=m, temperature=t),
        )

        registry.register(
            "UE",
            lambda c, m, t: UniformEnsemble(num_units=c, model_name=m, temperature=t),
        )

        # Judge operators
        registry.register(
            "J", lambda c, m, t: JudgeSynthesis(model_name=m, temperature=t)
        )

        registry.register(
            "JF", lambda c, m, t: JudgeSynthesis(model_name=m, temperature=t)
        )

        # Verifier
        registry.register("V", lambda c, m, t: Verifier(model_name=m, temperature=t))

        # Most Common (aggregation)
        registry.register("MC", lambda _, __, ___: MostCommon())

        # Prompt Enhancemer
        registry.register(
            "PE", lambda c, m, t: PromptEnhancer(model_name=m, temperature=t)
        )

        return registry


# Lazy initialization for default registry
_default_registry = None


def get_default_registry() -> OpRegistry:
    """Get the default operator registry, initializing it if needed.

    Returns:
        The default operator registry with standard types
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = OpRegistry.create_standard_registry()
    return _default_registry


def parse_spec(
    spec_str: str, registry: Optional[OpRegistry] = None
) -> Optional[Operator]:
    """Parse operator specification string into an operator instance.

    Format: "count:type:model:temperature"
    Example: "3:E:gpt-4o:0.7"

    Args:
        spec_str: Compact operator specification string
        registry: Optional registry to use (uses default if None)

    Returns:
        Instantiated operator, or None for reference specs (starting with $)

    Raises:
        ValueError: If the specification format is invalid
    """
    # Use the provided registry or get the default
    op_registry = registry or get_default_registry()

    # Reference - handled by resolve_refs
    if spec_str.startswith("$"):
        return None

    # Parse standard specification
    pattern = r"^(\d+):([A-Z]+):([^:]*):([0-9.]+)$"
    match = re.match(pattern, spec_str)

    if not match:
        raise ValueError(
            f"Invalid operator specification: '{spec_str}'. "
            f"Format must be 'count:type:model:temperature'"
        )

    count_str, type_code, model, temp_str = match.groups()

    try:
        count = int(count_str)
        temp = float(temp_str)
    except ValueError:
        raise ValueError(
            f"Invalid numeric values in '{spec_str}'. "
            f"Count must be an integer, temperature must be a float."
        )

    # Validate count must be positive
    if count <= 0:
        raise ValueError(f"Count must be a positive integer, got {count}")

    # Note: Temperature validation happens at the provider layer where
    # provider-specific constraints can be applied

    return op_registry.create(type_code, count, model, temp)


def resolve_refs(
    node_spec: NodeSpec,
    components: Optional[Dict[str, Any]] = None,
    type_registry: Optional[OpRegistry] = None,
) -> Operator:
    """Resolve operator references to concrete instances.

    The node can be:
    - A string operator specification ("3:E:gpt-4o:0.7")
    - A reference to a named component ("$name")
    - A list of nodes representing a sequential chain
    - An existing operator instance (passed through)

    Args:
        node_spec: Node specification
        components: Dictionary of named components for $references
        type_registry: Registry of operator type factories

    Returns:
        Resolved operator instance

    Raises:
        ValueError: If the node specification is invalid
        KeyError: If a referenced component is not found
    """
    # Use empty dict if no components provided
    local_components = components or {}

    # Already an operator instance
    if isinstance(node_spec, Operator):
        return node_spec

    # String format
    if isinstance(node_spec, str):
        # Component reference ($name)
        if node_spec.startswith("$"):
            name = node_spec[1:]  # Remove the $ prefix
            if name not in local_components:
                raise KeyError(
                    f"Referenced component '{name}' not found in component map"
                )

            target = local_components[name]
            # Recursively resolve if reference points to another spec
            return resolve_refs(target, local_components, type_registry)

        # Operator specification string
        return parse_spec(node_spec, type_registry)

    # List of nodes (sequential chain)
    if isinstance(node_spec, list):
        operators = [
            resolve_refs(node, local_components, type_registry) for node in node_spec
        ]
        return Sequential(operators=operators)

    # Invalid node type
    raise TypeError(f"Unsupported node type: {type(node_spec).__name__}")


def build_graph(
    graph_spec: GraphSpec,
    components: Optional[Dict[str, Any]] = None,
    type_registry: Optional[OpRegistry] = None,
) -> Operator:
    """Build an operator graph from a compact specification.

    Args:
        graph_spec: Graph specification (string, list, or operator)
        components: Dictionary of named components for $references
        type_registry: Registry of operator type factories

    Returns:
        Composed operator graph

    Raises:
        ValueError: If specification is invalid

    Examples:
        # Basic usage
        build_graph(["3:E:gpt-4o:0.7", "1:J:claude-3-5-sonnet:0.0"])

        # With named components
        components = {
            "sub": ["2:E:gpt-4o:0.0", "1:V:gpt-4o:0.0"]  # Reusable component
        }
        build_graph(["$sub", "$sub", "1:J:claude-3-5-sonnet:0.0"], components=components)

        # With custom type registry
        custom_registry = OpRegistry.create_standard_registry()
        custom_registry.register("CE", lambda c, m, t: CustomEnsemble(...))
        build_graph(["5:CE:gpt-4o:0.7"], type_registry=custom_registry)

        # With nested structure
        build_graph([
            ["3:E:gpt-4o:0.7", "1:V:gpt-4o:0.0"],  # Branch 1
            ["3:E:claude-3-5:0.7", "1:V:claude-3-5:0.0"],  # Branch 2
            "1:J:claude-3-5-sonnet:0.0"  # Final judge
        ])
    """
    # Process the graph with specified components and type registry
    return resolve_refs(graph_spec, components, type_registry)


# Public API
__all__ = [
    # Core graph building function
    "build_graph",
    # Operator registry for type extension
    "OpRegistry",
]
