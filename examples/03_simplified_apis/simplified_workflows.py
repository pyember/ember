"""Build workflows with minimal boilerplate.

This example demonstrates:
- Sequential workflows
- Parallel execution patterns
- Conditional workflows
- Error-tolerant workflows

Run with:
    python examples/03_simplified_apis/simplified_workflows.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ember.api import op


# =============================================================================
# Part 1: Sequential Workflows
# =============================================================================

@op
def step_1_validate(data: dict) -> dict:
    """Validate input data."""
    required = ["text", "options"]
    missing = [k for k in required if k not in data]
    if missing:
        return {"valid": False, "error": f"Missing: {missing}"}
    return {"valid": True, "data": data}


@op
def step_2_process(validated: dict) -> dict:
    """Process validated data."""
    if not validated.get("valid"):
        return validated  # Pass through errors

    text = validated["data"]["text"]
    return {
        "valid": True,
        "result": text.upper(),
        "length": len(text),
    }


@op
def step_3_format(processed: dict) -> dict:
    """Format the final output."""
    if not processed.get("valid"):
        return {"success": False, "error": processed.get("error", "Unknown")}

    return {
        "success": True,
        "output": processed["result"],
        "metadata": {"length": processed["length"]},
    }


def run_sequential(data: dict) -> dict:
    """Run steps in sequence."""
    result = step_1_validate(data)
    result = step_2_process(result)
    result = step_3_format(result)
    return result


# =============================================================================
# Part 2: Parallel Workflows
# =============================================================================

@op
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment (simulated)."""
    positive = ["good", "great", "excellent", "happy"]
    negative = ["bad", "terrible", "awful", "sad"]
    text_lower = text.lower()

    pos = sum(1 for w in positive if w in text_lower)
    neg = sum(1 for w in negative if w in text_lower)

    if pos > neg:
        sentiment = "positive"
    elif neg > pos:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"analysis": "sentiment", "result": sentiment}


@op
def analyze_length(text: str) -> dict:
    """Analyze text length."""
    words = text.split()
    return {
        "analysis": "length",
        "words": len(words),
        "characters": len(text),
    }


@op
def analyze_complexity(text: str) -> dict:
    """Analyze text complexity."""
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    return {
        "analysis": "complexity",
        "avg_word_length": round(avg_word_len, 2),
        "score": "complex" if avg_word_len > 6 else "simple",
    }


def run_parallel(text: str) -> dict:
    """Run multiple analyses in parallel."""
    analyzers = [analyze_sentiment, analyze_length, analyze_complexity]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fn, text) for fn in analyzers]
        results = [f.result() for f in futures]

    return {
        "text": text[:50] + "..." if len(text) > 50 else text,
        "analyses": {r["analysis"]: r for r in results},
    }


# =============================================================================
# Part 3: Conditional Workflows
# =============================================================================

@op
def route_by_length(text: str) -> dict:
    """Route to different processors based on input length."""
    word_count = len(text.split())

    if word_count < 10:
        return {"route": "short", "processor": process_short}
    elif word_count < 100:
        return {"route": "medium", "processor": process_medium}
    else:
        return {"route": "long", "processor": process_long}


@op
def process_short(text: str) -> dict:
    """Process short text."""
    return {"type": "short", "result": text.upper()}


@op
def process_medium(text: str) -> dict:
    """Process medium text."""
    words = text.split()
    return {"type": "medium", "result": " ".join(words[:20]) + "..."}


@op
def process_long(text: str) -> dict:
    """Process long text."""
    words = text.split()
    return {"type": "long", "result": f"Processed {len(words)} words"}


def run_conditional(text: str) -> dict:
    """Run conditional workflow based on routing."""
    routing = route_by_length(text)
    processor: Callable = routing["processor"]
    result = processor(text)
    return {"route": routing["route"], **result}


# =============================================================================
# Part 4: Error-Tolerant Workflows
# =============================================================================

def run_with_fallback(
    primary_fn: Callable,
    fallback_fn: Callable,
    *args,
    **kwargs,
) -> dict:
    """Run primary function, fall back on failure."""
    try:
        return {"source": "primary", "result": primary_fn(*args, **kwargs)}
    except Exception as e:
        return {"source": "fallback", "result": fallback_fn(*args, **kwargs), "error": str(e)}


def main() -> None:
    """Demonstrate simplified workflows."""
    print("Simplified Workflows")
    print("=" * 50)
    print()

    # Part 1: Sequential
    print("Part 1: Sequential Workflow")
    print("-" * 50)
    result = run_sequential({"text": "hello world", "options": {}})
    print(f"Valid input:   {result}")

    result = run_sequential({"text": "incomplete"})  # Missing 'options'
    print(f"Invalid input: {result}")
    print()

    # Part 2: Parallel
    print("Part 2: Parallel Workflow")
    print("-" * 50)
    sample = "This is a great example of parallel processing in Ember!"
    result = run_parallel(sample)
    print(f"Parallel analyses:")
    for name, analysis in result["analyses"].items():
        print(f"  {name}: {analysis}")
    print()

    # Part 3: Conditional
    print("Part 3: Conditional Workflow")
    print("-" * 50)
    texts = [
        "Short text.",
        "This is a medium length text that has more words.",
        " ".join(["word"] * 150),  # Long text
    ]
    for text in texts:
        result = run_conditional(text)
        print(f"  Route: {result['route']}, Type: {result['type']}")
    print()

    # Part 4: Error-tolerant
    print("Part 4: Error-Tolerant Workflow")
    print("-" * 50)

    def risky_fn(x: int) -> int:
        if x < 0:
            raise ValueError("Negative input")
        return x * 2

    def safe_fn(x: int) -> int:
        return abs(x)

    for x in [5, -3]:
        result = run_with_fallback(risky_fn, safe_fn, x)
        print(f"  Input {x}: {result}")
    print()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Chain operators for sequential workflows")
    print("2. Use ThreadPoolExecutor for parallel execution")
    print("3. Route dynamically based on input characteristics")
    print("4. Build error-tolerant pipelines with fallbacks")
    print()
    print("Next: See model_binding_patterns.py for model configuration")


if __name__ == "__main__":
    main()
