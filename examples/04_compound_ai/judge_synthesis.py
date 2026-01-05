"""Use judge models for quality control and synthesis.

This example demonstrates:
- Judge pattern for evaluating outputs
- Multi-step synthesis with verification
- Quality scoring and filtering
- Self-consistency checking

Run with:
    python examples/04_compound_ai/judge_synthesis.py
"""

from __future__ import annotations

from dataclasses import dataclass

from ember.api import op


# =============================================================================
# Part 1: Basic Judge Pattern
# =============================================================================

@dataclass
class JudgmentResult:
    """Result from a judge evaluation."""

    score: float  # 0.0 to 1.0
    passed: bool
    reasoning: str
    criteria_scores: dict[str, float]


@op
def judge_quality(
    content: str,
    criteria: list[str],
    threshold: float = 0.7,
) -> JudgmentResult:
    """Evaluate content quality against criteria.

    This is a heuristic implementation. In production,
    use an LLM as the judge for nuanced evaluation.
    """
    criteria_scores: dict[str, float] = {}

    for criterion in criteria:
        if criterion == "length":
            # Score based on reasonable length
            words = len(content.split())
            score = min(1.0, words / 50) if words < 200 else max(0.0, 1.0 - (words - 200) / 200)
        elif criterion == "clarity":
            # Simple heuristic: shorter sentences = clearer
            sentences = content.split(".")
            avg_words = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            score = max(0.0, 1.0 - (avg_words - 15) / 30)
        elif criterion == "completeness":
            # Check for key structural elements
            has_intro = any(w in content.lower() for w in ["is", "are", "the"])
            has_detail = len(content) > 100
            score = (0.5 if has_intro else 0.0) + (0.5 if has_detail else 0.0)
        else:
            score = 0.5  # Default for unknown criteria

        criteria_scores[criterion] = max(0.0, min(1.0, score))

    overall_score = sum(criteria_scores.values()) / max(len(criteria_scores), 1)
    passed = overall_score >= threshold

    return JudgmentResult(
        score=overall_score,
        passed=passed,
        reasoning=f"Overall score: {overall_score:.2f}, threshold: {threshold}",
        criteria_scores=criteria_scores,
    )


# =============================================================================
# Part 2: Generate-Judge-Refine Loop
# =============================================================================

@op
def generate_content(prompt: str, style: str = "detailed") -> str:
    """Generate content (simulated for this example)."""
    # In production, call an LLM
    if style == "brief":
        return f"Brief response to: {prompt[:30]}..."
    return f"Detailed response covering the topic of '{prompt[:30]}...' with comprehensive analysis."


@op
def refine_content(content: str, feedback: str) -> str:
    """Refine content based on feedback (simulated)."""
    # In production, call an LLM with the feedback
    return content + f" [Refined based on: {feedback[:20]}...]"


def generate_judge_refine(
    prompt: str,
    criteria: list[str],
    max_iterations: int = 3,
) -> dict:
    """Generate content with iterative refinement.

    Uses judge feedback to improve quality.
    """
    content = generate_content(prompt)
    history = []

    for iteration in range(max_iterations):
        judgment = judge_quality(content, criteria)
        history.append({
            "iteration": iteration,
            "score": judgment.score,
            "passed": judgment.passed,
        })

        if judgment.passed:
            break

        # Generate feedback for refinement
        low_criteria = [k for k, v in judgment.criteria_scores.items() if v < 0.7]
        feedback = f"Improve: {', '.join(low_criteria)}"
        content = refine_content(content, feedback)

    return {
        "final_content": content,
        "final_score": judgment.score,
        "iterations": len(history),
        "history": history,
    }


# =============================================================================
# Part 3: Multi-Response Synthesis
# =============================================================================

@op
def synthesize_best(responses: list[str], criteria: list[str]) -> dict:
    """Select the best response from multiple candidates.

    Judge each response and select the winner.
    """
    judgments = []

    for i, response in enumerate(responses):
        judgment = judge_quality(response, criteria)
        judgments.append({
            "index": i,
            "response": response[:50] + "...",
            "score": judgment.score,
            "passed": judgment.passed,
        })

    # Sort by score
    judgments.sort(key=lambda x: -x["score"])

    return {
        "best_index": judgments[0]["index"],
        "best_score": judgments[0]["score"],
        "rankings": judgments,
    }


# =============================================================================
# Part 4: Self-Consistency Check
# =============================================================================

@op
def check_consistency(responses: list[str]) -> dict:
    """Check if multiple responses are consistent.

    Useful for validating reasoning or factual answers.
    """
    if not responses:
        return {"consistent": False, "reason": "No responses"}

    # Simple consistency: check if responses have similar content
    # In production, use semantic similarity or LLM comparison
    first_words = set(responses[0].lower().split()[:10])

    agreements = 0
    for response in responses[1:]:
        response_words = set(response.lower().split()[:10])
        overlap = len(first_words & response_words) / max(len(first_words), 1)
        if overlap > 0.3:
            agreements += 1

    consistency_ratio = (agreements + 1) / len(responses)

    return {
        "consistent": consistency_ratio > 0.5,
        "consistency_ratio": consistency_ratio,
        "num_responses": len(responses),
        "agreements": agreements + 1,
    }


def main() -> None:
    """Demonstrate judge-synthesis patterns."""
    print("Judge-Synthesis Patterns")
    print("=" * 50)
    print()

    # Part 1: Basic judgment
    print("Part 1: Basic Judge Pattern")
    print("-" * 50)
    sample = "Machine learning enables computers to learn from data. It is widely used in modern applications."
    criteria = ["length", "clarity", "completeness"]
    judgment = judge_quality(sample, criteria)
    print(f"Content: {sample[:50]}...")
    print(f"Score: {judgment.score:.2f}")
    print(f"Passed: {judgment.passed}")
    print(f"Criteria: {judgment.criteria_scores}")
    print()

    # Part 2: Generate-Judge-Refine
    print("Part 2: Generate-Judge-Refine Loop")
    print("-" * 50)
    result = generate_judge_refine(
        "Explain neural networks",
        criteria=["length", "clarity"],
        max_iterations=2,
    )
    print(f"Final score: {result['final_score']:.2f}")
    print(f"Iterations: {result['iterations']}")
    print(f"History: {result['history']}")
    print()

    # Part 3: Multi-response synthesis
    print("Part 3: Multi-Response Synthesis")
    print("-" * 50)
    responses = [
        "Brief answer.",
        "A more detailed answer with additional context and explanation of the topic.",
        "Medium length response covering the key points.",
    ]
    result = synthesize_best(responses, ["length", "completeness"])
    print(f"Best response index: {result['best_index']}")
    print(f"Best score: {result['best_score']:.2f}")
    print()

    # Part 4: Self-consistency
    print("Part 4: Self-Consistency Check")
    print("-" * 50)
    responses = [
        "The answer is 42.",
        "42 is the answer.",
        "I believe the answer is 42.",
    ]
    result = check_consistency(responses)
    print(f"Consistent: {result['consistent']}")
    print(f"Ratio: {result['consistency_ratio']:.2f}")
    print()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Use judges to evaluate output quality")
    print("2. Iterate with feedback to improve results")
    print("3. Synthesize best response from multiple candidates")
    print("4. Check consistency for reliability")
    print()
    print("Next: See specifications_progressive.py for building specs")


if __name__ == "__main__":
    main()
