"""Enhanced Ensemble Example

This example demonstrates how to use an ensemble of PromptEnhancer operators
to generate multiple enhanced versions of a user query, and then synthesize
the best enhanced query using a JudgeSynthesis operator.

Workflow:
    1. The PromptEnhancer operator expands and clarifies the original query.
    2. An ensemble of language models generates multiple alternative enhanced queries.
    3. The JudgeSynthesis operator evaluates all enhanced queries and selects the best one.

To run:
    uv run python src/ember/examples/basic/enhanced_ensemble_example.py
"""

from ember.core import non

# Build a clear, step-by-step pipeline:
#   1. Enhance the original query for clarity and detail.
#   2. Generate 4 alternative enhanced queries using an ensemble of language models.
#   3. Judge and synthesize the best enhanced query from the ensemble outputs.

enhanced_ensemble = non.Sequential(
    operators=[
        # Step 1: Expand and clarify the user's query using a prompt enhancer.
        non.PromptEnhancer(
            model_name="openai:gpt-4.1-mini",
            temperature=0.8
        ),
        # Step 2: Generate 4 alternative enhanced queries in parallel.
        non.UniformEnsemble(
            num_units=4,
            model_name="openai:gpt-4.1",
            temperature=1.0
        ),
        # Step 3: Judge and synthesize the best enhanced query from the ensemble.
        non.JudgeSynthesis(
            model_name="openai:gpt-4.1",
            temperature=1.0
        ),
    ]
)

query = "What causes climate change?"
result = enhanced_ensemble(inputs={"query": query})

print("=== Enhanced Ensemble Example ===\n")
print(f"Original Query:\n  {query}\n")
print("Best Enhanced Query:")
if hasattr(result, "final_answer"):
    print(result.final_answer)
elif hasattr(result, "query"):
    print(result.query)
else:
    print(result)
