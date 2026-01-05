"""Build operators with progressive complexity disclosure.

This example demonstrates:
- Starting with simple functions
- Adding features incrementally
- Maintaining backwards compatibility
- Clean abstraction layers

Run with:
    python examples/04_compound_ai/operators_progressive_disclosure.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ember.api import op


# =============================================================================
# Part 1: Level 1 - Simple Functions
# =============================================================================

@op
def summarize_v1(text: str) -> str:
    """Simplest possible summarizer.

    Level 1: Just works, no configuration needed.
    """
    words = text.split()
    if len(words) <= 20:
        return text
    return " ".join(words[:20]) + "..."


# =============================================================================
# Part 2: Level 2 - Add Parameters
# =============================================================================

@op
def summarize_v2(text: str, max_words: int = 20, ellipsis: bool = True) -> str:
    """Summarizer with configuration options.

    Level 2: Optional parameters for customization.
    Defaults match v1 behavior for backwards compatibility.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    return truncated + "..." if ellipsis else truncated


# =============================================================================
# Part 3: Level 3 - Structured Input/Output
# =============================================================================

@dataclass
class SummaryRequest:
    """Structured input for summarization."""

    text: str
    max_words: int = 20
    style: str = "truncate"  # truncate, extract, abstract
    preserve_sentences: bool = False


@dataclass
class SummaryResult:
    """Structured output from summarization."""

    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    truncated: bool


@op
def summarize_v3(request: SummaryRequest) -> SummaryResult:
    """Summarizer with structured I/O.

    Level 3: Full control with typed specifications.
    """
    text = request.text
    words = text.split()
    original_length = len(words)

    if original_length <= request.max_words:
        return SummaryResult(
            summary=text,
            original_length=original_length,
            summary_length=original_length,
            compression_ratio=1.0,
            truncated=False,
        )

    # Apply style
    if request.style == "extract" and request.preserve_sentences:
        # Extract first complete sentences
        sentences = text.split(".")
        summary_words: list[str] = []
        for sentence in sentences:
            sentence_words = sentence.split()
            if len(summary_words) + len(sentence_words) <= request.max_words:
                summary_words.extend(sentence_words)
            else:
                break
        summary = " ".join(summary_words)
    else:
        summary = " ".join(words[: request.max_words]) + "..."

    summary_length = len(summary.split())
    return SummaryResult(
        summary=summary,
        original_length=original_length,
        summary_length=summary_length,
        compression_ratio=summary_length / original_length,
        truncated=True,
    )


# =============================================================================
# Part 4: Level 4 - Composable Pipeline
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    preprocess: Callable[[str], str] | None = None
    postprocess: Callable[[str], str] | None = None
    max_words: int = 20
    metadata: dict = field(default_factory=dict)


class SummaryPipeline:
    """Composable summarization pipeline.

    Level 4: Full composition with hooks and middleware.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def process(self, text: str) -> dict:
        """Run the full pipeline."""
        # Preprocess
        if self.config.preprocess:
            text = self.config.preprocess(text)

        # Core summarization
        request = SummaryRequest(text=text, max_words=self.config.max_words)
        result = summarize_v3(request)

        # Postprocess
        summary = result.summary
        if self.config.postprocess:
            summary = self.config.postprocess(summary)

        return {
            "summary": summary,
            "original_length": result.original_length,
            "compression_ratio": result.compression_ratio,
            "metadata": self.config.metadata,
        }


# =============================================================================
# Demonstration
# =============================================================================

def main() -> None:
    """Demonstrate progressive disclosure pattern."""
    print("Progressive Disclosure in Operators")
    print("=" * 50)
    print()

    sample_text = (
        "Machine learning is a subset of artificial intelligence. "
        "It enables computers to learn from data without explicit programming. "
        "Neural networks are a popular machine learning technique. "
        "They are inspired by the structure of the human brain."
    )

    # Level 1: Just works
    print("Level 1: Simple Function")
    print("-" * 50)
    result = summarize_v1(sample_text)
    print(f"Input:  {sample_text[:60]}...")
    print(f"Output: {result}")
    print()

    # Level 2: With parameters
    print("Level 2: With Parameters")
    print("-" * 50)
    result = summarize_v2(sample_text, max_words=10, ellipsis=True)
    print(f"max_words=10: {result}")
    result = summarize_v2(sample_text, max_words=10, ellipsis=False)
    print(f"ellipsis=False: {result}")
    print()

    # Level 3: Structured I/O
    print("Level 3: Structured I/O")
    print("-" * 50)
    request = SummaryRequest(text=sample_text, max_words=15, style="truncate")
    result = summarize_v3(request)
    print(f"Summary: {result.summary}")
    print(f"Compression: {result.compression_ratio:.1%}")
    print(f"Truncated: {result.truncated}")
    print()

    # Level 4: Full pipeline
    print("Level 4: Composable Pipeline")
    print("-" * 50)
    config = PipelineConfig(
        preprocess=lambda t: t.lower(),
        postprocess=lambda t: t.upper(),
        max_words=10,
        metadata={"source": "demo"},
    )
    pipeline = SummaryPipeline(config)
    result = pipeline.process(sample_text)
    print(f"Summary: {result['summary']}")
    print(f"Metadata: {result['metadata']}")
    print()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Start simple - complexity is opt-in")
    print("2. Add parameters without breaking existing code")
    print("3. Use dataclasses for structured I/O at higher levels")
    print("4. Compose with hooks/middleware for full control")
    print()
    print("Next: See judge_synthesis.py for quality control")


if __name__ == "__main__":
    main()
