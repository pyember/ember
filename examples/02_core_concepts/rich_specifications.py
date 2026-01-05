"""Use type specifications for input/output validation.

This example demonstrates:
- Pydantic models for structured I/O
- Automatic validation on operator calls
- Rich error messages for invalid inputs
- Schema generation for documentation

Run with:
    python examples/02_core_concepts/rich_specifications.py
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ember.api import op


# =============================================================================
# Part 1: Define Structured Types
# =============================================================================

class AnalysisRequest(BaseModel):
    """Input specification for text analysis."""

    text: str = Field(..., min_length=1, description="Text to analyze")
    language: str = Field(default="en", description="ISO language code")
    include_sentiment: bool = Field(default=True)


class AnalysisResult(BaseModel):
    """Output specification for text analysis."""

    word_count: int = Field(..., ge=0)
    char_count: int = Field(..., ge=0)
    language: str
    sentiment: Optional[Literal["positive", "negative", "neutral"]] = None


# =============================================================================
# Part 2: Operators with Type Specifications
# =============================================================================

@op
def analyze_with_spec(request: AnalysisRequest) -> AnalysisResult:
    """Analyze text using typed input/output.

    The @op decorator enables validation when input_spec/output_spec
    are set. This function uses Pydantic models directly.
    """
    text = request.text
    words = text.split()

    # Simple sentiment heuristic (in production, use an LLM)
    sentiment = None
    if request.include_sentiment:
        positive_words = {"good", "great", "excellent", "happy", "love"}
        negative_words = {"bad", "terrible", "awful", "sad", "hate"}
        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

    return AnalysisResult(
        word_count=len(words),
        char_count=len(text),
        language=request.language,
        sentiment=sentiment,
    )


# =============================================================================
# Part 3: Validation in Action
# =============================================================================

def demonstrate_validation() -> None:
    """Show how validation catches errors early."""
    print("Validation Examples")
    print("-" * 50)

    # Valid input
    valid_request = AnalysisRequest(
        text="This is a great example of Ember!",
        language="en",
        include_sentiment=True,
    )
    result = analyze_with_spec(valid_request)
    print(f"Valid input result: {result}")
    print()

    # Invalid input - empty text
    print("Attempting empty text (should fail validation):")
    try:
        invalid_request = AnalysisRequest(text="")
    except Exception as e:
        print(f"  Caught: {type(e).__name__}")
        print(f"  Message: String should have at least 1 character")
    print()

    # Dict input - Pydantic coerces automatically
    print("Dict input (Pydantic coerces to model):")
    dict_input = {"text": "Hello from a dictionary!", "language": "en"}
    request_from_dict = AnalysisRequest(**dict_input)
    result = analyze_with_spec(request_from_dict)
    print(f"  Result: {result}")
    print()


# =============================================================================
# Part 4: Schema Introspection
# =============================================================================

def show_schemas() -> None:
    """Display the JSON schemas for documentation."""
    print("Schema Introspection")
    print("-" * 50)

    print("Input Schema (AnalysisRequest):")
    schema = AnalysisRequest.model_json_schema()
    for field, props in schema.get("properties", {}).items():
        required = field in schema.get("required", [])
        print(f"  {field}: {props.get('type', 'any')}", end="")
        if required:
            print(" (required)", end="")
        if "default" in props:
            print(f" = {props['default']}", end="")
        print()
    print()

    print("Output Schema (AnalysisResult):")
    schema = AnalysisResult.model_json_schema()
    for field, props in schema.get("properties", {}).items():
        print(f"  {field}: {props.get('type', 'any')}")
    print()


def main() -> None:
    """Demonstrate rich specifications."""
    print("Rich Specifications with Pydantic")
    print("=" * 50)
    print()

    demonstrate_validation()
    show_schemas()

    print("Key Takeaways")
    print("-" * 50)
    print("1. Use Pydantic models for structured I/O")
    print("2. Validation happens automatically on construction")
    print("3. Rich error messages help debugging")
    print("4. Schemas enable documentation and tooling")
    print()
    print("Next: See context_management.py for configuration")


if __name__ == "__main__":
    main()
