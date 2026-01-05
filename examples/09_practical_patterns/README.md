# Practical Patterns

Real-world patterns for building LLM applications.

## Overview

These examples show common application patterns:
- Chain-of-thought reasoning
- Retrieval-augmented generation (RAG)
- Structured output extraction

## Examples

1. **chain_of_thought.py** - Step-by-Step Reasoning
   - Guide models through complex problems
   - Improve accuracy on reasoning tasks
   - Capture intermediate steps

2. **rag_pattern.py** - Retrieval-Augmented Generation
   - Combine retrieval with generation
   - Ground responses in source documents
   - Build knowledge-powered applications

3. **structured_output.py** - Extract Structured Data
   - Parse model output into typed objects
   - Validate and transform responses
   - Build reliable data pipelines

## Key Patterns

### Chain of Thought

```python
from ember.api import models

def solve_with_reasoning(problem: str) -> dict:
    prompt = f"""Solve this step by step:

    Problem: {problem}

    Think through each step, then give your final answer."""

    response = models("gpt-4o", prompt)
    return {"reasoning": response, "problem": problem}
```

### RAG Pattern

```python
def answer_with_context(question: str, documents: list[str]) -> str:
    context = "\n".join(documents)
    prompt = f"""Answer based on these documents:

    {context}

    Question: {question}
    Answer:"""

    return models("gpt-4o-mini", prompt)
```

### Structured Output

```python
import json

def extract_entities(text: str) -> dict:
    prompt = f"""Extract entities from this text as JSON:

    Text: {text}

    Format: {{"people": [...], "places": [...], "dates": [...]}}"""

    response = models("gpt-4o-mini", prompt, temperature=0)
    return json.loads(response)
```

## Prerequisites

Requires configured model providers.

## Next Steps

- **10_evaluation_suite/** - Evaluate your patterns systematically
- Review earlier examples to combine patterns effectively
