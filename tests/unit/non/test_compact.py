import pytest

from ember.non import (
    DEFAULT_NODE_REGISTRY,
    NodeRole,
    NodeType,
    build_graph,
    describe_graph,
    parse_flow_spec,
    parse_node_spec,
)


def test_parse_new_format_produces_node_id_and_params():
    node = parse_node_spec("5E@openai/gpt-4o(temp=0.7)", index=0)
    assert node.id == "e1"
    assert node.type_code == "E"
    assert node.count == 5
    assert node.model_id == "openai/gpt-4o"
    assert node.params["temperature"] == pytest.approx(0.7)


def test_parse_alias_overrides_node_id_and_is_preserved():
    node = parse_node_spec("2E!vote@openai/gpt-4o", index=0)
    assert node.id == "vote"
    assert node.alias == "vote"
    assert node.replicated_models() == ["openai/gpt-4o", "openai/gpt-4o"]


def test_parse_legacy_format_is_supported():
    node = parse_node_spec("7:E:openai:gpt-4o:0.5", index=1)
    assert node.id == "e2"
    assert node.count == 7
    assert node.model_id == "openai/gpt-4o"
    assert node.params["temperature"] == pytest.approx(0.5)


def test_build_graph_expands_components():
    graph = build_graph(
        ["$branch", "1J@anthropic/claude"],
        components={"branch": ["3E@openai/gpt-4o(temp=0.7)"]},
    )
    assert [node.type_code for node in graph.nodes] == ["E", "J"]
    assert graph.nodes[0].count == 3
    assert graph.flows  # sequential fallback


def test_build_graph_accepts_string_spec_and_components():
    spec_text = """
    $fanout
    1J@anthropic/claude-3-5-sonnet
    """
    component_text = """
    1E@openai/gpt-4o(temp=0.7)
    1M
    """
    flows_text = """
    e1 -> m2
    m2 -> j3
    """

    graph = build_graph(spec_text, components={"fanout": component_text}, flows=flows_text)

    assert [node.type_code for node in graph.nodes] == ["E", "M", "J"]
    assert graph.flows[0].original == "e1 -> m2"
    assert graph.flows[1].original == "m2 -> j3"
    assert graph.expanded_components["fanout"] == (
        "1E@openai/gpt-4o(temp=0.7)",
        "1M",
    )
    assert graph.original_spec == (
        "$fanout",
        "1J@anthropic/claude-3-5-sonnet",
    )


def test_describe_graph_returns_readable_summary():
    summary = describe_graph(
        "cheap_vote",
        [
            "5E@openai/gpt-4o-mini(temp=0.7)",
            "1M",
        ],
    )
    assert "Graph 'cheap_vote'" in summary
    assert "5" in summary
    assert "openai/gpt-4o-mini" in summary


def test_to_execution_plan_classifies_roles():
    graph = build_graph(
        [
            "3E@openai/gpt-4o(temp=0.6)",
            "1M",
            "1J@anthropic/claude-3-5-sonnet",
            "1V@anthropic/claude-3-5-haiku(temp=0.0)",
        ]
    )
    plan = graph.to_execution_plan()
    assert len(plan.candidates) == 1
    assert plan.aggregator is not None
    assert plan.judge is not None
    assert plan.verifier is not None


def test_unknown_node_type_errors_with_helpful_message():
    with pytest.raises(ValueError, match="Unknown NON operator type"):
        build_graph(["1X@openai/gpt-4o"])


def test_missing_required_model_errors():
    with pytest.raises(ValueError, match="requires a model identifier"):
        build_graph(["2E"])


def test_build_graph_rejects_duplicate_aliases():
    with pytest.raises(ValueError, match="Duplicate NON node identifier"):
        build_graph(
            [
                "1E!vote@openai/gpt-4o",
                "1J!vote@anthropic/claude-3-5-sonnet",
            ]
        )


def test_build_graph_validates_flow_endpoints():
    with pytest.raises(ValueError, match="Flow references unknown node"):
        build_graph(
            [
                "1E@openai/gpt-4o",
                "1J@anthropic/claude-3-5-sonnet",
            ],
            flows=["foo -> j2"],
        )


def test_custom_registry_supports_new_node_code():
    extra_type = NodeType(
        code="X",
        role=NodeRole.OTHER,
        description="Experimental operator",
        requires_model=False,
        allows_model=False,
        allowed_params=None,
    )
    registry = DEFAULT_NODE_REGISTRY.extend({"X": extra_type})
    graph = build_graph(["1X"], node_registry=registry)
    plan = graph.to_execution_plan()
    assert plan.candidates == ()


def test_parse_flow_spec_supports_ports_and_modifiers():
    spec = parse_flow_spec("e1.output, e2[] -> j1.verdict(append)")

    assert [endpoint.node for endpoint in spec.sources] == ["e1", "e2"]
    assert spec.sources[0].port == "output"
    assert spec.sources[1].modifier == "append"
    assert spec.targets[0].node == "j1"
    assert spec.targets[0].port == "verdict"
    assert spec.targets[0].modifier == "append"


def test_parse_node_spec_handles_complex_params():
    node = parse_node_spec(
        '1E@openai/gpt-4o(settings={"weights":[1,2]},flag=false,scale=1.5)',
        index=0,
    )

    assert node.params["settings"] == {"weights": [1, 2]}
    assert node.params["flag"] is False
    assert node.params["scale"] == pytest.approx(1.5)


