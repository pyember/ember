"""Graph Executor module providing data structures and execution routines for a node-on-node (NoN) graph.

This module defines classes and methods to build and execute graphs composed of interconnected
nodes, where each node is an operator that can generate or transform data.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.avior.registry.operator.operator_base import Operator
from src.avior.modules.lm_modules import LMModule, LMModuleConfig
from src.avior.registry.operator.operator_registry import OperatorRegistry

logger = logging.getLogger(__name__)

##############################################
# Graph Data Structures
##############################################


@dataclass
class GraphNode:
    """Represents a single node in a node-on-node (NoN) graph.

    Attributes:
        name: The unique name of this node.
        operator: An operator that processes data at this node.
        inputs: A list of node names whose outputs feed into this node's operator.
    """

    name: str
    operator: Operator
    inputs: List[str] = field(default_factory=list)


@dataclass
class NoNGraphData:
    """A pure data structure representing a node-on-node graph.

    This class holds a dictionary of GraphNode instances keyed by their names.
    It does not contain any execution logic.
    """

    nodes: Dict[str, GraphNode] = field(default_factory=dict)

    def add_node(self, name: str, operator: Operator, inputs: List[str]) -> None:
        """Adds a new node to the graph.

        Args:
            name: The unique identifier for the node.
            operator: The operator that will process data when this node is executed.
            inputs: A list of node names whose outputs feed into this node.

        Raises:
            ValueError: If a node with the given name already exists in the graph.
        """
        if name in self.nodes:
            raise ValueError(f"Node '{name}' already exists in the graph.")
        self.nodes[name] = GraphNode(name=name, operator=operator, inputs=inputs)


##############################################
# Graph Execution Core
##############################################


class GraphExecutor:
    """Executes a NoNGraphData by topologically sorting nodes and running each operator.

    This executor either:
    1. Handles a single-node graph directly, or
    2. Topologically sorts the graph and executes each node in sequence or in layers.
    Operators are responsible for handling how they execute (eagerly or via a plan-based approach).
    """

    def __init__(self, max_workers: int = None):
        """Initializes the GraphExecutor.

        Args:
            max_workers: The maximum number of worker threads to use for parallelized execution.
                If None, Python will choose an appropriate default value.
        """
        self.max_workers = max_workers

    def execute(
        self, graph_data: NoNGraphData, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs the graph to produce final outputs.

        If the graph has only one node, that node is immediately executed.
        Otherwise, the nodes are topologically sorted and executed layer by layer.

        Args:
            graph_data: The NoNGraphData instance representing the graph to execute.
            input_data: A dictionary of inputs required for graph execution.

        Returns:
            A dictionary containing the results from the final node of the graph.

        Raises:
            ValueError: If the graph has cycles or is not a valid directed acyclic graph (DAG).
        """
        if len(graph_data.nodes) == 1:
            single_node = next(iter(graph_data.nodes.values()))
            logger.debug("Single-node graph detected. Executing immediately.")
            return self._execute_node(
                node=single_node, results={}, input_data=input_data
            )

        logger.debug("Performing topological sort for multi-node graph.")
        sorted_nodes = self._topological_sort(graph_data=graph_data)
        results: Dict[str, Dict[str, Any]] = {}

        logger.debug("Starting layered execution plan.")
        for layer in self._layered_execution_plan(
            graph_data=graph_data, sorted_nodes=sorted_nodes
        ):
            layer_results = {}
            for node_name in layer:
                node = graph_data.nodes[node_name]
                logger.debug(f"Collecting inputs for node '{node_name}'.")
                node_input = self._collect_inputs(
                    node=node, results=results, input_data=input_data
                )
                node_result = self._execute_node(
                    node=node, results=results, input_data=node_input
                )
                layer_results[node_name] = node_result
            results.update(layer_results)

        final_node = sorted_nodes[-1]
        logger.debug(f"Returning results for final node '{final_node}'.")
        return results[final_node]

    def _execute_node(
        self,
        node: GraphNode,
        results: Dict[str, Dict[str, Any]],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Runs a single node in the graph.

        If the operator for the node is itself a NoNGraphData (i.e. a subgraph),
        this method will recursively execute that nested graph.

        Args:
            node: The GraphNode to execute.
            results: A dictionary of previous node results keyed by node names.
            input_data: A dictionary of input data required by this node.

        Returns:
            The result dictionary produced by the node's operator.
        """
        logger.debug(f"Executing node '{node.name}'.")
        if isinstance(node.operator, NoNGraphData):
            logger.debug(f"Operator for node '{node.name}' is a subgraph. Recursing.")
            sub_executor = GraphExecutor(max_workers=self.max_workers)
            return sub_executor.execute(graph_data=node.operator, input_data=input_data)
        return node.operator(**input_data)

    def _topological_sort(self, graph_data: NoNGraphData) -> List[str]:
        """Performs a topological sort on the provided NoNGraphData.

        Uses an in-degree approach: each node's inputs are used to compute
        how many dependencies it has. A zero in-degree means it can execute immediately.

        Args:
            graph_data: The NoNGraphData instance representing the graph to be sorted.

        Returns:
            A list of node names in topologically sorted order.

        Raises:
            ValueError: If a cycle is detected or if the sorted result does not include all nodes.
        """
        in_degree = {name: 0 for name in graph_data.nodes}
        for node in graph_data.nodes.values():
            for inp in node.inputs:
                in_degree[node.name] += 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        sorted_list = []

        while queue:
            v = queue.popleft()
            sorted_list.append(v)
            for w in [n for n in graph_data.nodes if v in graph_data.nodes[n].inputs]:
                in_degree[w] -= 1
                if in_degree[w] == 0:
                    queue.append(w)

        if len(sorted_list) != len(graph_data.nodes):
            logger.error("Graph has a cycle or is not a valid DAG.")
            raise ValueError("Graph has a cycle or is not a valid DAG.")
        return sorted_list

    def _layered_execution_plan(
        self, graph_data: NoNGraphData, sorted_nodes: List[str]
    ) -> List[List[str]]:
        """Produces a layered execution plan for the graph.

        This simple implementation assigns each node to its own layer, preserving
        topological order. More advanced strategies could group independent nodes
        together to enable parallel execution.

        Args:
            graph_data: The NoNGraphData containing the graph's nodes.
            sorted_nodes: A topologically sorted list of node names.

        Returns:
            A list of node layers, where each layer is a list of node names.
        """
        logger.debug("Generating a simple layer plan. Each node is its own layer.")
        return [[n] for n in sorted_nodes]

    def _collect_inputs(
        self,
        node: GraphNode,
        results: Dict[str, Dict[str, Any]],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregates all required inputs for a node from prior results and initial inputs.

        If an operator requires specific inputs (like "responses" or "query"), those
        values are combined from the input data and from the results of the nodes
        listed in `node.inputs`.

        Args:
            node: The current GraphNode being executed.
            results: A dictionary of results from nodes that have already executed.
            input_data: The initial input data provided to the graph executor.

        Returns:
            A dictionary containing all inputs required by this node's operator.
        """
        op = node.operator
        if isinstance(op, NoNGraphData):
            # If the operator is a subgraph, pass the entire input data set along
            logger.debug(
                f"Node '{node.name}' operator is a subgraph. Passing input data through."
            )
            return input_data

        required_inputs = (
            op.get_signature().required_inputs if op.get_signature() else []
        )
        combined = dict(input_data)

        if "responses" in required_inputs:
            logger.debug(f"Collecting 'responses' inputs for node '{node.name}'.")
            gathered_responses = []
            for inp in node.inputs:
                inp_result = results[inp]
                if "responses" in inp_result and isinstance(
                    inp_result["responses"], list
                ):
                    gathered_responses.extend(inp_result["responses"])
                elif "final_answer" in inp_result:
                    gathered_responses.append(inp_result["final_answer"])
            combined["responses"] = gathered_responses

        if "query" in required_inputs and "query" not in combined:
            logger.debug(
                f"'query' required but not present in combined inputs for node '{node.name}'."
            )
            if "query" in input_data:
                combined["query"] = input_data["query"]

        return combined


##############################################
# High-Level Execution Service
##############################################


class GraphExecutorService:
    """A high-level service that wraps the GraphExecutor to centralize logging, error handling,
    and potential expansion hooks for usage or model services.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize the GraphExecutorService with a core executor and a logger.

        Args:
            max_workers: Optional max worker limit for parallel execution.
        """
        self._executor_core = GraphExecutor(max_workers=max_workers)
        self._logger = logging.getLogger(self.__class__.__name__)

    def run(
        self, graph_data: NoNGraphData, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs the graph and handles errors and logging similarly
        to how model_service.py or usage_service.py encapsulate behavior.

        Args:
            graph_data: The NoNGraphData instance representing the graph to execute.
            input_data: A dictionary of inputs required for graph execution.

        Returns:
            A dictionary of the final node's results.

        Raises:
            Exception: Re-raises any exception after logging an error,
            so the caller can handle it if needed.
        """
        self._logger.info("GraphExecutorService run invoked.")
        try:
            result = self._executor_core.execute(graph_data, input_data)
            self._logger.info("Graph execution completed successfully.")
            return result
        except Exception as e:
            self._logger.error(f"Graph execution failed: {str(e)}", exc_info=True)
            raise


##############################################
# NoNGraphBuilder (Dictionary-based)
##############################################


class NoNGraphBuilder:
    """Builds a NoNGraphData object from a dictionary-based specification.

    The provided dictionary is expected to have the following format:
    {
        "node_name": {
            "op": "ENSEMBLE",
            "params": {"model_name": "gpt-4o", "temperature": 1.0, "count": 3},
            "inputs": ["other_node", ...]
        },
        ...
    }
    """

    def __init__(self):
        """Initializes a NoNGraphBuilder with an empty NoNGraphData."""
        self._graph_data = NoNGraphData()

    def parse_graph(
        self, graph: Dict[str, Dict[str, Any]], **kwargs: Any
    ) -> NoNGraphData:
        """Parses the given dictionary into a NoNGraphData structure.

        Args:
            graph: A dictionary describing the nodes of a graph.
            **kwargs: Additional parameters that may be passed to the underlying build.

        Returns:
            A populated NoNGraphData object.
        """
        return self._build_graph(config=graph, **kwargs)

    def _build_graph(
        self, config: Dict[str, Dict[str, Any]], **kwargs: Any
    ) -> NoNGraphData:
        """Internal method that constructs NoNGraphData from a configuration.

        Args:
            config: The dictionary specifying each node's operator, parameters, and inputs.
            **kwargs: Unused in this basic implementation, but available for extension.

        Returns:
            A fully constructed NoNGraphData object.
        """
        for name, node_config in config.items():
            op_entry = node_config.get("op")
            params = node_config.get("params", {})
            inputs = node_config.get("inputs", [])

            operator_class = OperatorRegistry.get(op_entry)
            lm_modules = self._create_lm_modules_from_params(params=params)

            op_instance = operator_class(lm_modules=lm_modules)
            self._graph_data.add_node(name, op_instance, inputs)
        return self._graph_data

    def _create_lm_modules_from_params(self, params: Dict[str, Any]) -> List[LMModule]:
        """Instantiates LMModule objects based on the provided parameters.

        Automatically handles the 'count' parameter to create multiple instances of LMModule
        with the same config.

        Args:
            params: A dictionary of parameters that may include 'model_name', 'temperature',
                'max_tokens', 'persona', and 'count'.

        Returns:
            A list of LMModule objects. Returns an empty list if 'model_name' is missing.
        """
        if "model_name" not in params:
            return []
        uc = dict(params)
        count = uc.pop("count", 1)
        config_for_lm = {
            k: v
            for k, v in uc.items()
            if k in ["model_name", "temperature", "max_tokens", "persona"]
        }
        return [LMModule(LMModuleConfig(**config_for_lm)) for _ in range(count)]



##############################################
# DSL for Graph Building (Alternative Approach)
##############################################


class GraphNodeBuilder:
    """A builder for a single graph node within a GraphBuilder DSL.

    This builder sets an operator, parameters, and inputs for the node,
    and finally builds the node into the underlying NoNGraphData structure.
    """

    def __init__(self, graph_builder: "GraphBuilder", name: str):
        """Initializes a GraphNodeBuilder.

        Args:
            graph_builder: A reference to the GraphBuilder managing this node.
            name: The name of the node to be constructed.
        """
        self.graph_builder = graph_builder
        self.name = name
        self._op_code = None
        self._params = {}
        self._inputs = []

    def operator(self, op_code: str):
        """Specifies the operator code (identifier) for the node.

        Args:
            op_code: The operator code, e.g., 'ENSEMBLE' or 'JUDGE'.

        Returns:
            The current GraphNodeBuilder (for chaining).
        """
        self._op_code = op_code
        return self

    def params(self, **kwargs):
        """Updates the parameters for the operator.

        Args:
            **kwargs: Arbitrary keyword arguments representing operator parameters.

        Returns:
            The current GraphNodeBuilder (for chaining).
        """
        self._params.update(kwargs)
        return self

    def inputs(self, *nodes: str):
        """Specifies the input node names for this node.

        Args:
            *nodes: Variable number of strings representing node names.

        Returns:
            The current GraphNodeBuilder (for chaining).
        """
        self._inputs.extend(nodes)
        return self

    def build(self):
        """Builds the node into the parent GraphBuilder's NoNGraphData.

        Raises:
            ValueError: If no operator code has been specified.
        """
        if self._op_code is None:
            raise ValueError(f"Node '{self.name}' has no operator defined.")

        operator_class = OperatorRegistry.get(self._op_code)

        lm_modules = []
        if "model_name" in self._params:
            count = self._params.pop("count", 1)
            config_for_lm = {
                k: v
                for k, v in self._params.items()
                if k in ["model_name", "temperature", "max_tokens", "persona"]
            }
            config_for_lm.setdefault("model_name", "gpt-4-turbo")
            config_for_lm.setdefault("temperature", 1.0)
            for _ in range(count):
                lm_modules.append(LMModule(LMModuleConfig(**config_for_lm)))

        op_instance = operator_class(lm_modules=lm_modules)
        self.graph_builder._graph_data.add_node(self.name, op_instance, self._inputs)


class GraphBuilder:
    """A DSL-based builder for constructing node-on-node graphs.

    This approach allows chaining commands for each node in a fluid style
    to produce a final NoNGraphData object.
    """

    def __init__(self):
        """Initializes a new GraphBuilder with an empty graph data structure."""
        self._graph_data = NoNGraphData()

    def node(self, name: str) -> GraphNodeBuilder:
        """Begins construction of a new node by name.

        Args:
            name: The node's name.

        Returns:
            A GraphNodeBuilder for configuring the node.
        """
        return GraphNodeBuilder(self, name)

    def build(self) -> NoNGraphData:
        """Finalizes the DSL build process and returns the constructed graph.

        Returns:
            A NoNGraphData object containing all configured nodes.
        """
        return self._graph_data


##############################################
# Example Usage
##############################################

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#
#     # Use the DSL
#     builder = GraphBuilder()
#     builder.node("ensemble1") \
#         .operator("ENSEMBLE") \
#         .params(model_name="gpt-4o", count=3) \
#         .build()
#
#     builder.node("ensemble2") \
#         .operator("ENSEMBLE") \
#         .params(model_name="gpt-4-turbo", count=2) \
#         .build()
#
#     builder.node("final_judge") \
#         .operator("JUDGE") \
#         .params(model_name="gpt-4o") \
#         .inputs("ensemble1", "ensemble2") \
#         .build()
#
#     graph = builder.build()
#
#     service = GraphExecutorService(max_workers=4)
#     input_data = {"query": "What is the capital of France?"}
#
#     # High-level service entry point
#     result = service.run(graph_data=graph, input_data=input_data)
#     logging.info(f"Final result from DSL-built graph: {result}")
