"""Showcase Ember's natural, Pythonic API design.

This example demonstrates:
- Simple, intuitive function calls
- Method chaining and fluent interfaces
- Sensible defaults that just work
- Progressive disclosure of complexity

Run with:
    python examples/03_simplified_apis/natural_api_showcase.py
"""

from __future__ import annotations

from ember.api import models, op


# =============================================================================
# Part 1: The Simplest Possible API
# =============================================================================

def demonstrate_simple_api() -> None:
    """Show the most basic usage patterns."""
    print("Part 1: Simple API")
    print("-" * 50)

    # Model discovery - just works
    available = models.list()[:5]
    print(f"Available models: {available}")

    # Get providers
    providers = models.providers()
    print(f"Providers: {providers}")
    print()


# =============================================================================
# Part 2: Natural Function Patterns
# =============================================================================

@op
def summarize(text: str, max_words: int = 50) -> str:
    """Summarize text to a maximum word count.

    This is the pattern: define what you want, not how to do it.
    The implementation can use LLMs or heuristics.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


@op
def extract_keywords(text: str, top_n: int = 5) -> list[str]:
    """Extract key terms from text.

    Simple heuristic version - in production, use an LLM.
    """
    # Simple heuristic: longer words that appear more than once
    words = text.lower().split()
    word_counts: dict[str, int] = {}
    for word in words:
        word = word.strip(".,!?;:")
        if len(word) > 4:  # Skip short words
            word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by count, return top N
    sorted_words = sorted(word_counts.items(), key=lambda x: -x[1])
    return [word for word, count in sorted_words[:top_n]]


@op
def analyze_document(text: str) -> dict:
    """High-level document analysis.

    Composes simpler operators naturally.
    """
    return {
        "summary": summarize(text, max_words=20),
        "keywords": extract_keywords(text, top_n=3),
        "word_count": len(text.split()),
        "char_count": len(text),
    }


# =============================================================================
# Part 3: Fluent Patterns
# =============================================================================

class TextProcessor:
    """Fluent interface for text processing.

    Allows method chaining: processor.clean().extract().summarize()
    """

    def __init__(self, text: str):
        self._text = text
        self._results: dict = {"original": text}

    def clean(self) -> "TextProcessor":
        """Remove extra whitespace and normalize."""
        self._text = " ".join(self._text.split())
        self._results["cleaned"] = self._text
        return self

    def extract_keywords(self, top_n: int = 5) -> "TextProcessor":
        """Extract keywords from current text."""
        self._results["keywords"] = extract_keywords(self._text, top_n)
        return self

    def summarize(self, max_words: int = 50) -> "TextProcessor":
        """Summarize current text."""
        self._results["summary"] = summarize(self._text, max_words)
        return self

    def result(self) -> dict:
        """Return accumulated results."""
        return self._results


# =============================================================================
# Part 4: Progressive Disclosure
# =============================================================================

def demonstrate_progressive_disclosure() -> None:
    """Show how API complexity is revealed progressively."""
    print("Part 4: Progressive Disclosure")
    print("-" * 50)

    # Level 1: Just call the function
    print("Level 1 - Simple call:")
    result = summarize("This is a simple text that needs summarization.")
    print(f"  {result}")
    print()

    # Level 2: Add parameters
    print("Level 2 - With parameters:")
    result = summarize("This is a longer text...", max_words=5)
    print(f"  {result}")
    print()

    # Level 3: Compose functions
    print("Level 3 - Composition:")
    text = "Machine learning is transforming how we build software applications."
    analysis = analyze_document(text)
    print(f"  Keywords: {analysis['keywords']}")
    print(f"  Summary: {analysis['summary']}")
    print()


def main() -> None:
    """Demonstrate natural API patterns."""
    print("Natural API Showcase")
    print("=" * 50)
    print()

    # Part 1: Simple API
    demonstrate_simple_api()

    # Part 2: Natural functions
    print("Part 2: Natural Functions")
    print("-" * 50)
    sample_text = (
        "Ember provides a natural API for building AI applications. "
        "The framework emphasizes simplicity and composability. "
        "Developers can build complex pipelines using simple functions."
    )
    analysis = analyze_document(sample_text)
    print(f"Document analysis: {analysis}")
    print()

    # Part 3: Fluent interface
    print("Part 3: Fluent Interface")
    print("-" * 50)
    result = (
        TextProcessor("  This   is   messy   text   with   extra   spaces.  ")
        .clean()
        .extract_keywords(top_n=3)
        .summarize(max_words=10)
        .result()
    )
    print(f"Fluent result: {result}")
    print()

    # Part 4: Progressive disclosure
    demonstrate_progressive_disclosure()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Start simple - complexity is opt-in")
    print("2. Functions read like descriptions of intent")
    print("3. Fluent interfaces enable readable chains")
    print("4. Sensible defaults minimize configuration")
    print()
    print("Next: See simplified_workflows.py for workflow patterns")


if __name__ == "__main__":
    main()
