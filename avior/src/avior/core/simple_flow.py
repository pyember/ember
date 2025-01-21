import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from avior.registry.operators.operator_registry import OperatorFactory, OperatorRegistry
from avior.registry.operators.operator_base import (
    Operator,
    NoNModule,
    OperatorContext,
    SingleUnitOperator,
)
from avior.core.configs.config import get_config


# ===================================
# ====== Operator Composition =======
# ===================================
@dataclass
class NoNGraph(NoNModule):
    graph: Optional[Union[List[Union[List, str]], Dict[str, Dict]]] = None

    def __post_init__(self):
        super().__init__()
        self.execution_order: List[str] = []
        self.node_inputs: Dict[str, List[str]] = {}
        self.graph_kwargs: Dict[str, Any] = {}  # Storing graph-specific kwargs

    def __hash__(self) -> int:
        graph_hash = (
            tuple(self.graph)
            if isinstance(self.graph, list)
            else frozenset(self.graph.items()) if self.graph else ()
        )
        return hash((graph_hash, frozenset(self.graph_kwargs.items())))

    def add_node(
        self,
        name: str,
        operation: Union[Callable, "NoNModule", "Operator", str, List[Any]],
        inputs: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        logging.debug(
            "Adding node: %s, operation: %s, inputs: %s", name, operation, inputs
        )

        try:
            parsed_operation = self._parse_operation(operation, **kwargs)
        except ValueError as e:
            logging.error("Failed to parse operation for node '%s': %s", name, e)
            raise ValueError(f"Invalid operation for node '{name}': {str(e)}") from e

        self._add_parsed_node(name, parsed_operation, inputs)

    def _parse_operation(
        self, operation: Any, **kwargs: Any
    ) -> Union[Callable, "NoNModule", "Operator", str]:
        if isinstance(operation, str):
            if ":" in operation:
                # Parsing component string (e.g., "3:E:gpt-4:1.0")
                parts = operation.split(":")
                if len(parts) > 1 and not self._validate_operator_code(parts[1]):
                    raise ValueError(f"Unknown operator code: {parts[1]}")
                return self.parse_component(operation, **kwargs)
            else:
                # This is an input node reference, validate it
                if not self._validate_operator_code(operation):
                    raise ValueError(f"Invalid input node reference: {operation}")
                return operation
        elif isinstance(operation, list):
            return NoNGraph().parse_from_list(operation, **kwargs)
        elif callable(operation) or isinstance(operation, (NoNModule, Operator)):
            return operation
        else:
            raise ValueError(f"Unknown operation type: {type(operation)}")

    def _add_parsed_node(
        self,
        name: str,
        operation: Union[Callable, "NoNModule", "Operator", str],
        inputs: Optional[List[str]] = None,
    ) -> None:
        if isinstance(operation, str):
            # This is an input node reference
            if not self._validate_operator_code(operation):
                raise ValueError(f"Invalid input node reference: {operation}")
            self.node_inputs[name] = [operation]
        else:
            self.add_module(name, operation)
            self.execution_order.append(name)
            self.node_inputs[name] = inputs or []

        logging.debug(
            "Node '%s' added successfully with operation '%s' and inputs '%s'",
            name,
            operation,
            inputs,
        )

    def forward(self, input_data: OperatorContext) -> OperatorContext:
        if not self.execution_order:
            logging.warning(
                "Execution order is empty. Returning empty OperatorContext."
            )
            return OperatorContext(final_answer=None)

        results: Dict[str, Any] = {}
        intermediate_results: Dict[str, Any] = {}

        for i, name in enumerate(self.execution_order):
            operation = self._modules[name]
            inputs = self.node_inputs.get(name, [])

            # Determine operation inputs
            if not inputs:
                operation_input = input_data
            else:
                operation_input = self._collect_inputs(inputs, results, input_data)

            result = operation(operation_input, **self.graph_kwargs)

            # Store results
            if isinstance(result, OperatorContext):
                results[name] = result.final_answer
                intermediate_results[name] = result.intermediate_results
            else:
                results[name] = result
                intermediate_results[name] = result

            logging.debug("Node '%s' executed. Result: %s", name, results[name])

        final_result = results[self.execution_order[-1]]
        logging.info(
            "Final result computed by node '%s': %s",
            self.execution_order[-1],
            final_result,
        )
        return OperatorContext(
            final_answer=final_result, intermediate_results=intermediate_results
        )

    def _collect_inputs(
        self, inputs: List[str], results: Dict[str, Any], input_data: OperatorContext
    ) -> OperatorContext:
        operation_inputs = [results.get(input_name) for input_name in inputs]
        if None in operation_inputs:
            missing_inputs = [
                inputs[i] for i, val in enumerate(operation_inputs) if val is None
            ]
            raise ValueError(f"No results available for inputs: {missing_inputs}")

        if len(operation_inputs) == 1:
            operation_input = operation_inputs[0]
        else:
            operation_input = operation_inputs

        return self._format_input(operation_input, input_data)

    def _format_input(
        self, operation_input: Any, original_input: OperatorContext
    ) -> OperatorContext:
        if isinstance(operation_input, OperatorContext):
            return operation_input

        responses = (
            [operation_input]
            if not isinstance(operation_input, list)
            else operation_input
        )

        return OperatorContext(
            query=original_input.query,
            context=original_input.context,
            choices=original_input.choices,
            responses=responses,
            metadata=original_input.metadata,
        )

    def parse_graph(
        self,
        graph: Union[List[Union[List[Any], Dict[str, Any]]], Dict[str, Dict[str, Any]]],
        named: bool = True,
        **kwargs: Any,
    ) -> "NoNGraph":
        self.graph = graph
        self.graph_kwargs = kwargs
        if named:
            if isinstance(graph, dict):
                return self._build_graph(graph, **kwargs)
            else:
                return self._parse_named_graph(graph, **kwargs)
        else:
            return self._parse_unnamed_graph(graph, **kwargs)

    def _build_graph(
        self, config: Dict[str, Dict[str, Any]], **kwargs: Any
    ) -> "NoNGraph":
        for name, node_config in config.items():
            op_code = node_config.get("op")
            params = node_config.get("params", {})
            prompt_name = params.pop("prompt_name", None)
            custom_prompts = params.pop("custom_prompts", None)

            op = OperatorFactory.create(
                op_code,
                [params],
                prompt_name=prompt_name,
                custom_prompts=custom_prompts,
                **kwargs,
            )
            inputs = node_config.get("inputs", [])
            self.add_node(name, op, inputs=inputs)
        return self

    def _parse_named_graph(
        self, graph: List[Union[List[Any], Dict[str, Any]]], **kwargs: Any
    ) -> "NoNGraph":
        for i, item in enumerate(graph):
            if isinstance(item, list):
                if len(item) == 3 and isinstance(item[1], str):
                    # This is a node specification
                    name, op_str, inputs = item
                    operation = self.parse_component(op_str, **kwargs)
                    self.add_node(name, operation, inputs, **kwargs)
                else:
                    # This is a nested subgraph
                    subgraph_name = f"subgraph_{i}"
                    sub_graph = NoNGraph()._parse_named_graph(item, **kwargs)
                    self.add_node(subgraph_name, sub_graph)
            elif isinstance(item, dict):
                # Support for explicitly named subgraphs
                if len(item) != 1:
                    raise ValueError(
                        f"Invalid named subgraph specification: {item}. "
                        "Expected a dict with a single key-value pair."
                    )
                subgraph_name, subgraph_spec = next(iter(item.items()))
                sub_graph = NoNGraph()._parse_named_graph(subgraph_spec, **kwargs)
                self.add_node(subgraph_name, sub_graph)
            else:
                raise ValueError(
                    f"Invalid graph item: {item}. Expected a list or dict."
                )
        return self

    def _parse_unnamed_graph(
        self, graph: List[Union[List[Any], str]], **kwargs: Any
    ) -> "NoNGraph":
        # Initializing a list to keep track of node names in sequence
        node_sequence: List[str] = []

        for i, item in enumerate(graph):
            if isinstance(item, list):
                # Handling nested graph
                name = f"subgraph_{i}"
                sub_graph = NoNGraph()._parse_unnamed_graph(item, **kwargs)
                self.add_node(name, sub_graph)
                node_sequence.append(name)
            elif isinstance(item, str):
                # Handling regular node
                name = f"node_{i}"
                self.add_node(name, item, **kwargs)
                node_sequence.append(name)
            else:
                raise ValueError(f"Invalid item type in graph: {type(item)}")

        # Setting up node_inputs based on the sequence
        for i, node_name in enumerate(node_sequence):
            if i > 0:  # All nodes except the first one have inputs
                self.node_inputs[node_name] = node_sequence[:i]

        return self

    def parse_from_list(
        self, graph: List[Union[List[Any], str]], **kwargs: Any
    ) -> "NoNGraph":
        return self._parse_unnamed_graph(graph, **kwargs)

    def parse_component(
        self, comp_str: str, **kwargs: Any
    ) -> Union["Operator", Callable]:
        logging.debug("Parsing component: %s", comp_str)
        parts = comp_str.split(":")
        if not parts or len(parts) < 2:
            raise ValueError(f"Invalid component string: {comp_str}")

        count_str, op_type_code = parts[0], parts[1]
        try:
            count = int(count_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid count '{count_str}' in component string: {comp_str}"
            ) from e

        # Handle persona in op_type_code
        persona = self._extract_persona(op_type_code)
        op_type_code = op_type_code.split("|")[0]  # Remove persona if present

        # Getting the operator class to check if it's a SingleUnitOperator
        operator_entry = OperatorRegistry.get(op_type_code)
        if not operator_entry:
            raise ValueError(f"Unknown operator code: {op_type_code}")
        operator_class, _ = operator_entry

        # Creating a single config
        config = {
            "model_name": self._get_model_name(parts, **kwargs),
            "temperature": self._get_temperature(parts, **kwargs),
            "max_tokens": self._get_max_tokens(parts, **kwargs),
            "persona": persona
            or kwargs.get("persona")
            or get_config("personas", "default_persona", fallback=None),
        }

        logging.debug("Created config: %s", config)

        # For SingleUnitOperators, ignoring count and using a single config
        if issubclass(operator_class, SingleUnitOperator):
            unit_configs = [config]
        else:
            unit_configs = [config] * count

        return OperatorFactory.create(op_type_code, unit_configs)

    def _get_model_name(self, parts: List[str], **kwargs: Any) -> str:
        if len(parts) > 2:
            return parts[2]
        return kwargs.get(
            "model_name", get_config("models", "default_model", fallback="gpt-4-turbo")
        )

    def _get_temperature(self, parts: List[str], **kwargs: Any) -> float:
        if len(parts) > 3:
            return float(parts[3])
        return float(
            kwargs.get(
                "temperature", get_config("models", "default_temperature", fallback=1.0)
            )
        )

    def _get_max_tokens(self, parts: List[str], **kwargs: Any) -> Optional[int]:
        if len(parts) > 4:
            return int(parts[4])
        max_tokens = kwargs.get(
            "max_tokens", get_config("tokens", "default_max_tokens", fallback=None)
        )
        return int(max_tokens) if max_tokens is not None else None

    def _extract_persona(self, op_type_code: str) -> Optional[str]:
        if "|" in op_type_code:
            _, persona = op_type_code.split("|", 1)
            return persona
        return None

    def __str__(self) -> str:
        return f"NoNGraph(nodes={len(self.execution_order)})"

    def __repr__(self) -> str:
        return f"NoNGraph(execution_order={self.execution_order}, node_inputs={self.node_inputs})"

    def _validate_operator_code(self, op_code: str) -> bool:
        base_op_code = op_code.split("|")[0]  # Remove persona if present
        is_valid = OperatorRegistry.get(base_op_code) is not None
        logging.debug("Operator code '%s' validation result: %s", op_code, is_valid)
        return is_valid
