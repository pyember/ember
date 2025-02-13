"""
DSL-based builder for constructing an XCSGraph directly.

This module replaces the legacy GraphBuilder and GraphNodeBuilder.
"""

from typing import Any, Dict, List, Optional
import logging

from ember.core.registry.operator.operator_registry import OperatorRegistry
from ember.core.registry.operator.core.operator_base import (
    Operator,
    LMModule,
    LMModuleConfig,
)
from .xcs_graph import XCSGraph

logger = logging.getLogger(__name__)


class XCSGraphNodeBuilder:
    """Builder for configuring and constructing a single node within an XCSGraph.

    Attributes:
        graph_builder (XCSGraphBuilder): The parent graph builder instance responsible for assembling the graph.
        name (str): A unique identifier for the node.
        _op_code (Optional[str]): The operator code used to retrieve the operator class.
        _params (Dict[str, Any]): Additional parameters for operator instantiation.
        _inputs (List[str]): A list of node identifiers whose outputs will serve as inputs for this node.
    """

    def __init__(self, graph_builder: "XCSGraphBuilder", name: str) -> None:
        """Initializes the node builder.

        Args:
            graph_builder (XCSGraphBuilder): The parent graph builder instance.
            name (str): The unique identifier for the node.
        """
        self.graph_builder: "XCSGraphBuilder" = graph_builder
        self.name: str = name
        self._op_code: Optional[str] = None
        self._params: Dict[str, Any] = {}
        self._inputs: List[str] = []

    def operator(self, *, op_code: str) -> "XCSGraphNodeBuilder":
        """Specifies the operator for this node using its operator code.

        Args:
            op_code (str): The operator code to look up in the registry.

        Returns:
            XCSGraphNodeBuilder: The current builder instance for method chaining.
        """
        self._op_code = op_code
        return self

    def params(self, **kwargs: Any) -> "XCSGraphNodeBuilder":
        """Sets additional parameters for operator configuration.

        Args:
            **kwargs (Any): Arbitrary keyword arguments representing operator parameters.

        Returns:
            XCSGraphNodeBuilder: The current builder instance for method chaining.
        """
        self._params.update(kwargs)
        return self

    def inputs(self, *nodes: str) -> "XCSGraphNodeBuilder":
        """Specifies input nodes whose outputs will serve as inputs for this node.

        Args:
            *nodes (str): Variable length argument list of node identifiers.

        Returns:
            XCSGraphNodeBuilder: The current builder instance for method chaining.
        """
        self._inputs.extend(nodes)
        return self

    def build(self) -> None:
        """Constructs the node and integrates it into the parent XCSGraph.

        The operator is instantiated based on the provided operator code and parameters.
        If language model configuration is detected in the parameters, the necessary LM modules are created.
        After instantiation, the node is registered in the graph and directed edges are established
        from the specified input nodes.

        Returns:
            None

        Raises:
            ValueError: If no operator has been defined for the node or if the operator code is not found in the registry.
        """
        if self._op_code is None:
            raise ValueError(f"Node '{self.name}' has no operator defined.")

        operator_class = OperatorRegistry.get(self._op_code)
        if operator_class is None:
            raise ValueError(f"Operator code '{self._op_code}' not found in registry.")

        lm_modules: List[LMModule] = []
        if "model_name" in self._params:
            count: int = int(self._params.pop("count", 1))
            allowed_keys: List[str] = [
                "model_name",
                "temperature",
                "max_tokens",
                "persona",
            ]
            config_for_lm: Dict[str, Any] = {
                k: v for k, v in self._params.items() if k in allowed_keys
            }
            for _ in range(count):
                lm_config: LMModuleConfig = LMModuleConfig(**config_for_lm)
                lm_modules.append(LMModule(lm_config))

        op_instance: Operator = operator_class(lm_modules=lm_modules)
        self.graph_builder.graph.add_node(operator=op_instance, node_id=self.name)
        for input_node in self._inputs:
            self.graph_builder.graph.add_edge(from_id=input_node, to_id=self.name)


class XCSGraphBuilder:
    """DSL-based builder for constructing an XCSGraph.

    This builder provides a fluent interface for incrementally assembling an XCSGraph by adding and configuring nodes.

    Attributes:
        graph (XCSGraph): The XCSGraph instance being constructed.
    """

    def __init__(self) -> None:
        """Initializes a new, empty XCSGraph.

        The graph attribute accumulates nodes and edges during construction.

        Returns:
            None
        """
        self.graph: XCSGraph = XCSGraph()

    def node(self, *, name: str) -> XCSGraphNodeBuilder:
        """Starts the construction of a new node with the given unique identifier.

        Args:
            name (str): The unique identifier to assign to the new node.

        Returns:
            XCSGraphNodeBuilder: A builder instance to further configure the new node.
        """
        return XCSGraphNodeBuilder(graph_builder=self, name=name)

    def build(self) -> XCSGraph:
        """Finalizes and returns the constructed XCSGraph.

        Returns:
            XCSGraph: The fully constructed graph.
        """
        return self.graph
