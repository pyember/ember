import logging
from collections import deque
from typing import Any, Dict, List, Optional

from src.avior.core.trace_context import get_current_trace_context, TraceRecord
from .operator_graph import OperatorGraph

logger = logging.getLogger(__name__)


class OperatorGraphRunner:
    """
    Executes an OperatorGraph by:
      1. If there's only one node, run it immediately with the user-provided input_data.
      2. Otherwise, topologically sort the graph, 
         collect inputs from upstream nodes, and run each node in order.
      3. Return the outputs from the final (exit) node of the graph.
      
    This merges the old 'GraphExecutor' approach.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        max_workers can be used if we later implement concurrency or 
        partial parallelism. Currently, we run sequentially in this example.
        """
        self.max_workers = max_workers

    def run(self, graph: OperatorGraph, input_data: Dict[str, Any]) -> Any:
        """
        Execute the entire graph and return the final node's output.
        """
        node_ids = list(graph.nodes.keys())
        if len(node_ids) == 0:
            raise ValueError("OperatorGraph is empty; no nodes to run.")
        if len(node_ids) == 1:
            # Single node => run directly
            node_id = node_ids[0]
            logger.debug("Single-node graph detected. Executing immediately.")
            return self._execute_node(graph, node_id, {}, input_data)

        logger.debug("Multiple-node graph; performing topological sort.")
        sorted_ids = self._topological_sort(graph)
        results: Dict[str, Any] = {}

        for node_id in sorted_ids:
            # Collect relevant inputs from upstream results
            node_input = self._collect_inputs(graph, node_id, results, input_data)
            # Execute
            output_data = self._execute_node(graph, node_id, results, node_input)
            results[node_id] = output_data

        # Return the final node's output
        exit_node = graph.exit_node
        if not exit_node:
            # fallback => last sorted node if exit_node is None for some reason
            exit_node = sorted_ids[-1]
        return results[exit_node]

    def _execute_node(
        self,
        graph: OperatorGraph,
        node_id: str,
        results: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> Any:
        """
        Runs a single node in the graph with the given input_data.
        Stores the node's output in node.captured_outputs.
        Also does trace_context logging if active.
        """
        node = graph.get_node(node_id)
        logger.debug(f"Executing node '{node_id}'.")
        node.captured_inputs = dict(input_data)

        operator = node.operator
        output = operator(input_data)  # calls Operator.__call__, which includes forward()
        node.captured_outputs = output

        # If there's a TraceContext, record
        ctx = get_current_trace_context()
        if ctx:
            record = TraceRecord(
                operator_name=getattr(operator, "name", node_id),
                operator_class=operator.__class__.__name__,
                input_data=input_data,
                output_data=output,
            )
            ctx.add_record(record)

        return output

    def _collect_inputs(
        self,
        graph: OperatorGraph,
        node_id: str,
        results: Dict[str, Any],
        global_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregates all required inputs for this node from:
          - global_input
          - outputs of upstream nodes 
            (based on inbound_edges)
        If the Operator requires certain fields, we gather them. 
        Otherwise, we pass everything from global_input by default.
        """
        node = graph.get_node(node_id)
        op = node.operator

        # If no inbound edges => might use global_input only
        required_inputs = (
            op.get_signature().required_inputs
            if op.get_signature() else []
        )

        # Start with global_input as a base
        combined = dict(global_input)

        # Merge data from upstream nodes
        for parent_id in node.inbound_edges:
            parent_output = results.get(parent_id, {})
            # For certain fields like "responses", we gather them from parents
            # This logic can be extended or customized per your usage
            if "responses" in required_inputs:
                # gather parent's 'responses' or 'final_answer'
                if isinstance(parent_output, dict):
                    resp = parent_output.get("responses")
                    if isinstance(resp, list):
                        # extend or merge
                        if "responses" not in combined:
                            combined["responses"] = []
                        combined["responses"].extend(resp)
                    elif "final_answer" in parent_output:
                        if "responses" not in combined:
                            combined["responses"] = []
                        combined["responses"].append(parent_output["final_answer"])

            # Optionally handle "query" if missing
            if "query" in required_inputs and "query" not in combined:
                if "query" in parent_output:
                    combined["query"] = parent_output["query"]

        return combined

    def _topological_sort(self, graph: OperatorGraph) -> List[str]:
        """
        Performs a topological sort using in-degree from inbound_edges.
        """
        in_degree = {nid: 0 for nid in graph.nodes}
        for nid, node in graph.nodes.items():
            for p in node.inbound_edges:
                in_degree[nid] += 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        sorted_list = []

        while queue:
            current = queue.popleft()
            sorted_list.append(current)
            for out_id in graph.get_node(current).outbound_edges:
                in_degree[out_id] -= 1
                if in_degree[out_id] == 0:
                    queue.append(out_id)

        # check for cycles
        if len(sorted_list) != len(graph.nodes):
            raise ValueError("OperatorGraph has a cycle or is not a valid DAG.")
        return sorted_list
    
class OperatorGraphRunnerService:
    def __init__(self, max_workers: Optional[int] = None):
        self._runner = OperatorGraphRunner(max_workers=max_workers)

    def run(self, graph: OperatorGraph, input_data: Dict[str, Any]) -> Any:
        # do logging, error handling, then delegate
        try:
            return self._runner.run(graph, input_data)
        except Exception as e:
            # handle or re-raise
            raise