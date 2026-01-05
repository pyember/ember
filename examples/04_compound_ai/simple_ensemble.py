"""Build multi-model ensembles for improved reliability.

This example demonstrates:
- Calling multiple models for the same task
- Aggregating responses (voting, averaging)
- Ensemble strategies for robustness
- Cost-quality tradeoffs

Run with:
    python examples/04_compound_ai/simple_ensemble.py
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from ember.api import models, op


# =============================================================================
# Part 1: Basic Ensemble Pattern
# =============================================================================

@op
def call_model_safe(model_id: str, prompt: str) -> dict:
    """Call a model with error handling."""
    try:
        response = models(model_id, prompt)
        return {"model": model_id, "success": True, "response": response}
    except Exception as e:
        return {"model": model_id, "success": False, "error": str(e)}


def ensemble_call(prompt: str, model_ids: list[str]) -> list[dict]:
    """Call multiple models in parallel."""
    with ThreadPoolExecutor(max_workers=len(model_ids)) as executor:
        futures = [executor.submit(call_model_safe, m, prompt) for m in model_ids]
        return [f.result() for f in futures]


# =============================================================================
# Part 2: Voting Ensemble
# =============================================================================

@op
def voting_ensemble(
    prompt: str,
    models_list: list[str],
    extract_answer: Callable[[str], str],
) -> dict:
    """Use majority voting to select the best answer.

    Args:
        prompt: The question to ask
        models_list: List of model IDs to query
        extract_answer: Function to extract the votable answer from response

    Returns:
        Dict with winner and vote counts
    """
    responses = ensemble_call(prompt, models_list)

    # Extract answers and count votes
    votes: Counter = Counter()
    details = []

    for r in responses:
        if r["success"]:
            answer = extract_answer(r["response"])
            votes[answer] += 1
            details.append({"model": r["model"], "answer": answer})
        else:
            details.append({"model": r["model"], "error": r["error"]})

    if not votes:
        return {"winner": None, "confidence": 0, "details": details}

    winner, count = votes.most_common(1)[0]
    total = sum(votes.values())

    return {
        "winner": winner,
        "confidence": count / total,
        "vote_counts": dict(votes),
        "details": details,
    }


# =============================================================================
# Part 3: Weighted Ensemble
# =============================================================================

class WeightedEnsemble:
    """Ensemble with model-specific weights."""

    def __init__(self, weights: dict[str, float]):
        """Initialize with model weights (higher = more influence)."""
        self.weights = weights
        total = sum(weights.values())
        self.normalized = {k: v / total for k, v in weights.items()}

    def score_responses(self, responses: list[dict]) -> dict:
        """Score responses by weighted voting."""
        scores: dict[str, float] = {}

        for r in responses:
            if r["success"]:
                model = r["model"]
                answer = r["response"][:50]  # Truncate for comparison
                weight = self.normalized.get(model, 0.1)
                scores[answer] = scores.get(answer, 0) + weight

        if not scores:
            return {"best": None, "scores": {}}

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return {"best": best, "scores": scores}


# =============================================================================
# Part 4: Simulated Ensemble Demo
# =============================================================================

def simulate_model_response(model_id: str, prompt: str) -> dict:
    """Simulate model responses for demonstration without API keys."""
    # Simulated responses for different models
    simulated = {
        "model_a": "Paris",
        "model_b": "Paris",
        "model_c": "Lyon",  # Wrong answer
    }
    return {
        "model": model_id,
        "success": True,
        "response": simulated.get(model_id, "Unknown"),
    }


def demonstrate_voting_simulation() -> None:
    """Demonstrate voting with simulated responses."""
    print("Part 2: Voting Ensemble (Simulated)")
    print("-" * 50)

    responses = [
        simulate_model_response(m, "What is the capital of France?")
        for m in ["model_a", "model_b", "model_c"]
    ]

    votes: Counter = Counter()
    for r in responses:
        votes[r["response"]] += 1
        print(f"  {r['model']}: {r['response']}")

    winner, count = votes.most_common(1)[0]
    print(f"  Winner (majority): {winner} ({count}/3 votes)")
    print()


def demonstrate_weighted_simulation() -> None:
    """Demonstrate weighted ensemble with simulation."""
    print("Part 3: Weighted Ensemble (Simulated)")
    print("-" * 50)

    # Quality-based weights
    weights = {"model_a": 0.5, "model_b": 0.3, "model_c": 0.2}
    ensemble = WeightedEnsemble(weights)

    responses = [
        simulate_model_response(m, "What is 2+2?")
        for m in weights.keys()
    ]

    print(f"  Weights: {weights}")
    for r in responses:
        print(f"  {r['model']}: {r['response']}")

    result = ensemble.score_responses(responses)
    print(f"  Best (weighted): {result['best']}")
    print()


# =============================================================================
# Part 5: Ensemble Strategies
# =============================================================================

def demonstrate_strategies() -> None:
    """Show different ensemble strategies."""
    print("Part 4: Ensemble Strategies")
    print("-" * 50)

    strategies = {
        "Majority Vote": "Simple voting, best for classification",
        "Weighted Vote": "Models weighted by quality/cost",
        "Best-of-N": "Generate N responses, pick best",
        "Cascade": "Try cheap model first, escalate if needed",
        "Mixture-of-Experts": "Route to specialist models by task type",
    }

    for name, description in strategies.items():
        print(f"  {name}:")
        print(f"    {description}")
    print()


def main() -> None:
    """Demonstrate ensemble patterns."""
    print("Simple Ensemble Patterns")
    print("=" * 50)
    print()

    # Part 1: Overview
    print("Part 1: Ensemble Overview")
    print("-" * 50)
    print("Ensembles combine multiple models for:")
    print("  - Improved accuracy (wisdom of crowds)")
    print("  - Better reliability (redundancy)")
    print("  - Reduced variance (averaging)")
    print()

    # Parts 2-3: Simulated demonstrations
    demonstrate_voting_simulation()
    demonstrate_weighted_simulation()

    # Part 4: Strategies
    demonstrate_strategies()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Use parallel calls for efficiency")
    print("2. Majority voting works well for classification")
    print("3. Weight models by quality for better results")
    print("4. Consider cost-quality tradeoffs")
    print()
    print("Next: See operators_progressive_disclosure.py for composition")


if __name__ == "__main__":
    main()
