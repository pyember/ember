"""Understand Ember operators - the building blocks of AI applications.

This example demonstrates:
- Functions as operators with the @op decorator
- Operator composition and chaining
- Pure computation (no API keys required)
- Integration patterns with LLMs

Run with:
    python examples/02_core_concepts/operators_basics.py
"""

from __future__ import annotations

from ember.api import op


# =============================================================================
# Part 1: Basic Operators
# =============================================================================

@op
def clean_text(text: str) -> str:
    """Normalize text by stripping whitespace and lowercasing."""
    return text.strip().lower()


@op
def count_words(text: str, min_length: int = 1) -> dict:
    """Count words in text, optionally filtering by minimum length."""
    words = text.split()
    filtered = [w for w in words if len(w) >= min_length]
    return {
        "total": len(words),
        "filtered": len(filtered),
        "words": filtered[:5],  # Sample
    }


# =============================================================================
# Part 2: Operator Composition
# =============================================================================

@op
def analyze_text(text: str) -> dict:
    """Compose operators into a pipeline."""
    cleaned = clean_text(text)
    stats = count_words(cleaned, min_length=3)
    return {
        "original": text,
        "cleaned": cleaned,
        "stats": stats,
    }


# =============================================================================
# Part 3: Stateful Operators
# =============================================================================

@op
def classify_question(question: str) -> dict:
    """Classify a question by type and complexity.

    This demonstrates a more complex operator with internal logic.
    In practice, you might use an LLM for this classification.
    """
    question_lower = question.lower()

    # Determine question type
    if question_lower.startswith(("what", "who", "where", "when")):
        q_type = "factual"
    elif question_lower.startswith(("why", "how")):
        q_type = "explanatory"
    elif question_lower.startswith(("is", "are", "can", "do", "does")):
        q_type = "yes_no"
    else:
        q_type = "open_ended"

    # Estimate complexity by word count and clause indicators
    words = question.split()
    has_clauses = any(w in question_lower for w in ["and", "or", "but", "because"])
    complexity = "complex" if len(words) > 15 or has_clauses else "simple"

    return {
        "question": question,
        "type": q_type,
        "complexity": complexity,
    }


# =============================================================================
# Part 4: Batch Processing Pattern
# =============================================================================

def analyze_questions(questions: list[str]) -> list[dict]:
    """Process multiple questions.

    In production, use vmap() for automatic parallelization.
    This is the manual pattern for demonstration.
    """
    return [classify_question(q) for q in questions]


def main() -> None:
    """Demonstrate operator basics."""
    print("Ember Operators Basics")
    print("=" * 50)
    print()

    # Part 1: Basic operators
    print("Part 1: Basic Operators")
    print("-" * 50)

    sample = "  Hello WORLD! This is Ember.  "
    cleaned = clean_text(sample)
    print(f"Original: {repr(sample)}")
    print(f"Cleaned:  {repr(cleaned)}")

    stats = count_words("Learning to use Ember operators is fun", min_length=3)
    print(f"Word stats: {stats}")
    print()

    # Part 2: Composition
    print("Part 2: Operator Composition")
    print("-" * 50)

    result = analyze_text("  The Quick BROWN Fox Jumps!  ")
    print(f"Analysis: {result}")
    print()

    # Part 3: Classification
    print("Part 3: Question Classification")
    print("-" * 50)

    questions = [
        "What is machine learning?",
        "Why does gravity exist?",
        "Is Python good for AI?",
        "How do neural networks learn from data and improve over time?",
    ]

    for q in questions:
        result = classify_question(q)
        print(f"Q: {q}")
        print(f"   Type: {result['type']}, Complexity: {result['complexity']}")
    print()

    # Part 4: Batch processing
    print("Part 4: Batch Processing")
    print("-" * 50)

    results = analyze_questions(questions)
    print(f"Processed {len(results)} questions")
    print(f"Types: {[r['type'] for r in results]}")
    print()

    # Key takeaways
    print("Key Takeaways")
    print("-" * 50)
    print("1. @op decorator marks functions as operators")
    print("2. Operators compose naturally as Python functions")
    print("3. Pure operators work without API keys")
    print("4. Use vmap() for automatic batch parallelization")
    print()
    print("Next: See rich_specifications.py for type validation")


if __name__ == "__main__":
    main()
