import logging
from collections import deque
from typing import Any, Dict, List, Optional

from ember.src.ember.core.tracer.trace_context import (
    get_current_trace_context,
    TraceRecord,
)
from .operator_graph import OperatorGraph

logger: logging.Logger = logging.getLogger(__name__)


class OperatorGraphRunner:
    """Executes an OperatorGraph.

    The execution strategy is as follows:
      1. If the graph contains a single node, execute it immediately with the provided input data.
      2. If multiple nodes exist, perform a topological sort to determine execution order,
         collect required inputs from upstream nodes, and then execute each node in order.
      3. The output from the final (exit) node is returned.

    This class merges functionalities from the legacy 'GraphExecutor'.
    """

    def __init__(self, max_workers: Optional[int] = None) -> None:
        """Initializes an OperatorGraphRunner.

        Args:
            max_workers (Optional[int]): Maximum number of workers for future concurrency
                or partial parallelism. Currently, execution is sequential.
        """
        self.max_workers: Optional[int] = max_workers

    def run(self, *, graph: OperatorGraph, input_data: Dict[str, Any]) -> Any:
        """Executes the entire operator graph and returns the final node's output.

        Args:
            graph (OperatorGraph): The operator graph to execute.
            input_data (Dict[str, Any]): The initial input data for graph execution.

        Returns:
            Any: The output of the final node in the graph.

        Raises:
            ValueError: If the operator graph is empty.
        """
        node_ids: List[str] = list(graph.nodes.keys())
        if not node_ids:
            raise ValueError("OperatorGraph is empty; no nodes to run.")

        if len(node_ids) == 1:
            # Single-node graph: execute directly.
            node_id: str = node_ids[0]
            logger.debug("Single-node graph detected. Executing immediately.")
            return self._execute_node(
                graph=graph, node_id=node_id, results={}, input_data=input_data
            )

        logger.debug("Multiple-node graph detected. Performing topological sort.")
        sorted_ids: List[str] = self._topological_sort(graph=graph)
        results: Dict[str, Any] = {}

        for node_id in sorted_ids:
            # Collect inputs from upstream nodes and global input.
            node_input: Dict[str, Any] = self._collect_inputs(
                graph=graph,
                node_id=node_id,
                results=results,
                global_input=input_data,
            )
            # Execute this node.
            output_data: Any = self._execute_node(
                graph=graph, node_id=node_id, results=results, input_data=node_input
            )
            results[node_id] = output_data

        # Determine the exit node.
        exit_node: Optional[str] = graph.exit_node
        if not exit_node:
            # Fallback to the last node in the sorted order if exit_node is not defined.
            exit_node = sorted_ids[-1]
        return results[exit_node]

    def _execute_node(
        self,
        *,
        graph: OperatorGraph,
        node_id: str,
        results: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Any:
        """Executes a single node within the graph.

        Captures input and output data for debugging and tracing purposes.

        Args:
            graph (OperatorGraph): The operator graph.
            node_id (str): The identifier of the node to execute.
            results (Dict[str, Any]): A mapping of previously computed node results.
            input_data (Dict[str, Any]): The input data for the node's execution.

        Returns:
            Any: The output produced by executing the node.
        """
        node = graph.get_node(node_id=node_id)
        logger.debug("Executing node '%s'.", node_id)
        node.captured_inputs = dict(input_data)  # Preserve a copy of input data.

        operator = node.operator
        output: Any = operator(input_data=input_data)  # Named argument invocation.
        node.captured_outputs = output

        # If a trace context is available, record the execution details.
        ctx = get_current_trace_context()
        if ctx is not None:
            record = TraceRecord(
                operator_name=getattr(operator, "name", node_id),
                operator_class=operator.__class__.__name__,
                input_data=input_data,
                output_data=output,
            )
            ctx.add_record(record=record)
        return output

    def _collect_inputs(
        self,
        *,
        graph: OperatorGraph,
        node_id: str,
        results: Dict[str, Any],
        global_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregates required inputs for a node's execution from global input and upstream outputs.

        The aggregation strategy is as follows:
            - Start with the provided global input.
            - For each parent node (via inbound edges), merge outputs as needed.
              For example, if the operator requires 'responses', aggregate those from parent nodes.
            - If the operator requires specific fields (e.g., 'query') and they are missing,
              attempt to retrieve them from upstream outputs.

        Args:
            graph (OperatorGraph): The operator graph.
            node_id (str): The identifier of the node for which inputs are collected.
            results (Dict[str, Any]): A mapping of previously computed node results.
            global_input (Dict[str, Any]): The global input data provided to the graph.

        Returns:
            Dict[str, Any]: A dictionary containing the combined inputs for the node.
        """
        node = graph.get_node(node_id=node_id)
        operator = node.operator

        signature = operator.get_signature()
        required_inputs: List[str] = signature.required_inputs if signature else []

        # Begin with a copy of the global input.
        combined: Dict[str, Any] = dict(global_input)

        # Merge outputs from upstream nodes.
        for parent_id in node.inbound_edges:
            parent_output: Dict[str, Any] = results.get(parent_id, {})
            # For fields like "responses", combine responses from multiple parent nodes.
            if "responses" in required_inputs:
                if isinstance(parent_output, dict):
                    responses = parent_output.get("responses")
                    if isinstance(responses, list):
                        combined.setdefault("responses", []).extend(responses)
                    elif "final_answer" in parent_output:
                        combined.setdefault("responses", []).append(
                            parent_output["final_answer"]
                        )

            # Ensure 'query' is included if required.
            if "query" in required_inputs and "query" not in combined:
                if "query" in parent_output:
                    combined["query"] = parent_output["query"]

        return combined

    def _topological_sort(self, *, graph: OperatorGraph) -> List[str]:
        """Performs a topological sort of the operator graph based on inbound edges.

        Args:
            graph (OperatorGraph): The operator graph to sort.

        Returns:
            List[str]: A list of node identifiers in topologically sorted order.

        Raises:
            ValueError: If the graph contains a cycle or is not a valid Directed Acyclic Graph (DAG).
        """
        in_degree: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}
        for node_id, node in graph.nodes.items():
            for _ in node.inbound_edges:
                in_degree[node_id] += 1

        queue: deque[str] = deque(
            [node_id for node_id, degree in in_degree.items() if degree == 0]
        )
        sorted_list: List[str] = []

        while queue:
            current: str = queue.popleft()
            sorted_list.append(current)
            for adjacent in graph.get_node(node_id=current).outbound_edges:
                in_degree[adjacent] -= 1
                if in_degree[adjacent] == 0:
                    queue.append(adjacent)

        if len(sorted_list) != len(graph.nodes):
            raise ValueError("OperatorGraph has a cycle or is not a valid DAG.")
        return sorted_list


class OperatorGraphRunnerService:
    """Service layer for executing an OperatorGraph with additional logging and error handling."""

    def __init__(self, max_workers: Optional[int] = None) -> None:
        """Initializes the OperatorGraphRunnerService.

        Args:
            max_workers (Optional[int]): Maximum number of workers for future concurrency.
        """
        self._runner: OperatorGraphRunner = OperatorGraphRunner(max_workers=max_workers)

    def run(self, *, graph: OperatorGraph, input_data: Dict[str, Any]) -> Any:
        """Executes the operator graph and returns its final output with error handling.

        Args:
            graph (OperatorGraph): The operator graph to be executed.
            input_data (Dict[str, Any]): The input data to drive the graph's execution.

        Returns:
            Any: The output of the final node in the operator graph.

        Raises:
            Exception: Propagates any exception that occurs during graph execution.
        """
        try:
            return self._runner.run(graph=graph, input_data=input_data)
        except Exception as error:
            logger.exception("An error occurred while executing the operator graph.")
            raise error
