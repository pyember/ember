"""FastMCP server implementation for Ember - simplified setup."""

import asyncio
import json
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
except ImportError as exc:
    raise ImportError(
        "FastMCP is required for this integration. Install with: pip install fastmcp"
    ) from exc

from ember._internal.context import MetricsContext
from ember.api import models, operators

# Initialize FastMCP server
mcp = FastMCP("ember-mcp")
metrics_context = MetricsContext()


@mcp.tool()
async def ember_generate(
    prompt: str, model: str = "claude-3-haiku-20240307", temperature: float = 0.7
) -> str:
    """Generate text using any model in Ember's registry.

    Args:
        prompt: The input prompt
        model: Model identifier (e.g., 'claude-3-opus', 'gpt-4')
        temperature: Sampling temperature (0.0-1.0)

    Returns:
        Generated text response
    """
    try:
        ember_model = models.get_model(model)

        # Run generation with metrics tracking
        with metrics_context.track(model=model) as span:
            response = await asyncio.to_thread(
                ember_model.generate, prompt, temperature=temperature
            )
            span.record_response(response, model=model)

        # Extract content
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        else:
            return str(response)

    except Exception as e:
        return f"Error: Failed to generate with {model} - {str(e)}"


@mcp.tool()
async def ember_compare(prompt: str, models: List[str]) -> Dict[str, Any]:
    """Compare outputs from multiple models.

    Args:
        prompt: The input prompt
        models: List of model identifiers to compare

    Returns:
        Dictionary with model outputs and metrics
    """
    results = {}

    for model_name in models:
        try:
            output = await ember_generate(prompt, model_name)

            # Get metrics from last call
            last_metrics = metrics_context.get_last_metrics()

            results[model_name] = {
                "output": output,
                "tokens": last_metrics.get("usage", {}).get("total_tokens", 0),
                "latency_ms": last_metrics.get("latency_ms", 0),
            }
        except Exception as e:
            results[model_name] = {
                "output": f"Error: {str(e)}",
                "tokens": 0,
                "latency_ms": 0,
            }

    return results


@mcp.tool()
async def ember_ensemble(
    prompt: str, models: List[str], strategy: str = "majority_vote"
) -> Dict[str, Any]:
    """Run ensemble voting across multiple models.

    Args:
        prompt: The input prompt
        models: List of model identifiers
        strategy: Voting strategy ('majority_vote', 'weighted', 'confidence')

    Returns:
        Ensemble result with consensus and voting details
    """
    try:
        # Create operators for each model
        ops = [operators.Operator(model=m) for m in models]

        # Create ensemble
        ensemble = operators.EnsembleOperator(operators=ops, strategy=strategy)

        # Run ensemble
        result = await asyncio.to_thread(ensemble.run, prompt)

        return {
            "consensus": result.content,
            "votes": result.metadata.get("votes", {}),
            "confidence": result.metadata.get("confidence", 0.0),
            "models_used": models,
        }

    except Exception as e:
        return {"error": str(e), "consensus": None, "models_used": models}


@mcp.tool()
async def ember_code_review(
    code: str, language: str = "python", focus_areas: Optional[List[str]] = None
) -> str:
    """Perform comprehensive code review using Ember.

    Args:
        code: Code to review
        language: Programming language
        focus_areas: Specific areas to focus on (e.g., ['security', 'performance'])

    Returns:
        Detailed code review
    """
    # Default focus areas
    if focus_areas is None:
        focus_areas = [
            "correctness",
            "performance",
            "security",
            "style",
            "maintainability",
        ]

    prompt = f"""You are an expert {language} code reviewer. Review the following code focusing on:
{chr(10).join(f"- {area.capitalize()}" for area in focus_areas)}

Code to review:
```{language}
{code}
```

Provide a structured review with specific feedback and suggestions."""

    # Use GPT-4 for code review by default
    return await ember_generate(prompt, model="gpt-4", temperature=0.3)


@mcp.tool()
async def ember_explain(concept: str, audience: str = "general", style: str = "clear") -> str:
    """Explain a concept using Ember's models.

    Args:
        concept: Concept to explain
        audience: Target audience ('beginner', 'general', 'expert')
        style: Explanation style ('simple', 'clear', 'detailed', 'eli5')

    Returns:
        Clear explanation of the concept
    """
    style_prompts = {
        "simple": "Explain in simple terms",
        "clear": "Provide a clear, well-structured explanation",
        "detailed": "Give a comprehensive, detailed explanation",
        "eli5": "Explain like I'm 5 years old",
    }

    audience_context = {
        "beginner": "someone with no background knowledge",
        "general": "someone with general knowledge",
        "expert": "someone with deep expertise",
    }

    prompt = f"""{style_prompts.get(style, style_prompts["clear"])} of '{concept}' 
for {audience_context.get(audience, audience_context["general"])}.
Include relevant examples where helpful."""

    # Use Claude for explanations by default
    return await ember_generate(prompt, model="claude-3-sonnet-20240229", temperature=0.5)


# Resources
@mcp.resource("ember://models/list")
async def list_models() -> str:
    """List all available models in Ember's registry."""
    try:
        registry = models.get_registry()

        model_list = []
        for model_id, info in registry.items():
            model_list.append(
                {
                    "id": model_id,
                    "provider": info.provider,
                    "context_length": info.context_length,
                    "supports_tools": info.supports_tools,
                    "supports_streaming": info.supports_streaming,
                }
            )

        return json.dumps(model_list, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("ember://models/costs")
async def get_model_costs() -> str:
    """Get pricing information for all models."""
    try:
        registry = models.get_registry()

        costs = {}
        for model_id, info in registry.items():
            costs[model_id] = {
                "input_per_1m_tokens": f"${info.input_cost:.4f}",
                "output_per_1m_tokens": f"${info.output_cost:.4f}",
            }

        return json.dumps(costs, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.resource("ember://usage/current")
async def get_current_usage() -> str:
    """Get current session usage statistics."""
    try:
        # This would be replaced with real metrics in production
        usage = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "by_model": {},
            "session_duration_minutes": 0,
        }

        return json.dumps(usage, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# Prompts
@mcp.prompt()
async def debug_prompt(
    error_message: str, code_context: Optional[str] = None
) -> List[Dict[str, str]]:
    """Generate a debugging prompt for an error.

    Args:
        error_message: The error message to debug
        code_context: Optional code context where error occurred

    Returns:
        Messages for debugging the error
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert debugger. Analyze errors, identify root causes, "
                "and provide solutions."
            ),
        },
        {"role": "user", "content": f"I'm encountering this error:\n\n{error_message}"},
    ]

    if code_context:
        messages[-1]["content"] += f"\n\nCode context:\n```\n{code_context}\n```"

    messages[-1]["content"] += (
        "\n\nPlease help me:\n1. Understand what this error means"
        "\n2. Identify the likely cause\n3. Provide a solution"
    )

    return messages


@mcp.prompt()
async def optimization_prompt(
    code: str, optimization_goal: str = "performance"
) -> List[Dict[str, str]]:
    """Generate optimization suggestions for code.

    Args:
        code: Code to optimize
        optimization_goal: What to optimize for ('performance', 'memory', 'readability')

    Returns:
        Messages for optimization analysis
    """
    goals = {
        "performance": "execution speed and efficiency",
        "memory": "memory usage and allocation",
        "readability": "code clarity and maintainability",
    }

    return [
        {
            "role": "system",
            "content": (
                "You are an expert at code optimization, particularly for "
                f"{goals.get(optimization_goal, optimization_goal)}."
            ),
        },
        {
            "role": "user",
            "content": (
                "Please analyze this code and suggest optimizations for "
                f"{optimization_goal}:\n\n```\n{code}\n```\n\n"
                "Provide:\n"
                "1. Current inefficiencies\n"
                "2. Specific optimization suggestions\n"
                "3. Potential trade-offs\n"
                "4. Estimated impact\n"
                "5. Implementation steps"
            ),
        },
    ]


# Main entry point
if __name__ == "__main__":
    # Run the FastMCP server
    import sys

    # Check if running in stdio mode
    if len(sys.argv) > 1 and sys.argv[1] == "--stdio":
        mcp.run(transport="stdio")
    else:
        # Default to stdio
        mcp.run()
