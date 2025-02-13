"""Parser for JSON/dictionary specifications to build an XCSGraph.

This module substitutes the legacy NoNGraphBuilder and converts a specification
into an XCSGraph instance.

Expected Specification Format:
    {
        "node_name": {
            "op": "OPERATOR_CODE",
            "params": { ... },
            "inputs": ["other_node", ...]
        },
        ...
    }
"""

import copy
import logging
from typing import Any, Dict, List, Optional

from ember.core.registry.operator.operator_registry import OperatorRegistry
from ember.core.registry.operator.core.operator_base import (
    Operator,
    LMModule,
    LMModuleConfig,
)
from ..graph.xcs_graph import XCSGraph

logger = logging.getLogger(__name__)


def parse_spec_to_xcsgraph(spec: Dict[str, Dict[str, Any]]) -> XCSGraph:
    """Constructs an XCSGraph from a specification dictionary.

    The specification must adhere to the format:
        {
            "node_name": {
                "op": "OPERATOR_CODE",
                "params": { ... },
                "inputs": ["other_node", ...]
            },
            ...
        }

    Args:
        spec (Dict[str, Dict[str, Any]]): A dictionary mapping node names to their
            configuration. Each configuration must include an 'op' key and may optionally
            include 'params' and 'inputs' keys.

    Returns:
        XCSGraph: The graph built from the provided specification.

    Raises:
        ValueError: If a node configuration is missing the 'op' key or if the operator code
            is not found in the registry.
    """
    graph: XCSGraph = XCSGraph()

    # Map node names to their corresponding input lists.
    # Use a separate dictionary to prevent mutation of the input specification.
    node_inputs: Dict[str, List[str]] = {}

    # First Pass: Create nodes.
    for node_name, node_config in spec.items():
        op_code: Optional[str] = node_config.get("op")
        if op_code is None:
            raise ValueError(
                f"Configuration for node '{node_name}' must include an 'op' entry."
            )

        # Make a deep copy of the parameters to avoid side effects.
        params: Dict[str, Any] = copy.deepcopy(node_config.get("params", {}))
        inputs: List[str] = list(node_config.get("inputs", []))
        node_inputs[node_name] = inputs

        operator_class = OperatorRegistry.get(op_code)
        if operator_class is None:
            raise ValueError(
                f"Operator code '{op_code}' not found in registry for node '{node_name}'."
            )

        # Instantiate LMModules if a 'model_name' is provided in the parameters.
        lm_modules: List[LMModule] = []
        if "model_name" in params:
            count: int = int(params.pop("count", 1))
            allowed_keys = {"model_name", "temperature", "max_tokens", "persona"}
            config_for_lm: Dict[str, Any] = {
                key: value for key, value in params.items() if key in allowed_keys
            }
            lm_config: LMModuleConfig = LMModuleConfig(**config_for_lm)
            for _ in range(count):
                lm_modules.append(LMModule(lm_config))

        # Instantiate the operator and add the node to the graph.
        op_instance: Operator = operator_class(lm_modules=lm_modules)
        graph.add_node(operator=op_instance, node_id=node_name)

    # Second Pass: Create edges based on input mappings.
    for target_node, inputs_list in node_inputs.items():
        for source_node in inputs_list:
            graph.add_edge(from_id=source_node, to_id=target_node)

    return graph
