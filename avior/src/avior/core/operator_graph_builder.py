# operator_graph_builder.py
import uuid
from typing import Any, Dict, List, Optional
from src.avior.registry.operator.operator_registry import OperatorRegistry
from src.avior.registry.operator.operator_base import LMModule, LMModuleConfig, Operator
from .operator_graph import OperatorGraph


class ConfigGraphBuilder:
    """
    Builds an OperatorGraph from a dictionary specification, 
    similar to the old NoNGraphBuilder. The config format is:
        {
            "node_name": {
                "op": "ENSEMBLE", 
                "params": {"model_name": "gpt-4o", "count": 3},
                "inputs": ["other_node", ...]
            },
            ...
        }
    """

    def __init__(self):
        self._graph = OperatorGraph()
        # We keep a mapping from user-friendly names (e.g. "node_name") 
        # to internal node IDs (if you want them distinct). 
        # Or we can simply reuse "node_name" as the node_id. 
        self._name_to_id: Dict[str, str] = {}

    def parse_graph(self, graph_dict: Dict[str, Dict[str, Any]]) -> OperatorGraph:
        """
        Parse the dictionary definition into an OperatorGraph.
        Returns the constructed graph.
        """
        # 1) Create nodes
        for node_name, node_spec in graph_dict.items():
            node_id = node_name  # or something like str(uuid.uuid4())[:8]
            op_code = node_spec.get("op")
            params = node_spec.get("params", {})
            operator_cls = OperatorRegistry.get(op_code)

            if not operator_cls:
                raise ValueError(f"Operator code '{op_code}' not found in registry.")

            lm_modules = self._create_lm_modules_from_params(params)
            op_instance = operator_cls(lm_modules=lm_modules)

            new_id = self._graph.add_node(op_instance, node_id=node_id)
            self._name_to_id[node_name] = new_id

        # 2) Create edges
        for node_name, node_spec in graph_dict.items():
            inputs = node_spec.get("inputs", [])
            to_id = self._name_to_id[node_name]
            for inp_name in inputs:
                from_id = self._name_to_id[inp_name]
                self._graph.add_edge(from_id, to_id)

        return self._graph

    def _create_lm_modules_from_params(self, params: Dict[str, Any]) -> List[LMModule]:
        """
        Creates LMModule(s) from the config 'params' block, 
        handling optional 'count' for multiple modules.
        """
        if "model_name" not in params:
            return []
        config_copy = dict(params)
        count = config_copy.pop("count", 1)
        config_for_lm = {
            k: v for k, v in config_copy.items()
            if k in ["model_name", "temperature", "max_tokens", "persona"]
        }
        modules = []
        for _ in range(count):
            cfg = LMModuleConfig(**config_for_lm)
            modules.append(LMModule(cfg))
        return modules