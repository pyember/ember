"""
Example demonstrating the simplified import structure.

This example shows how to use the new top-level imports for operators and NON components.
"""

from ember.non import JudgeSynthesis, Sequential, UniformEnsemble

# Create an ensemble with 3 identical models
ensemble = UniformEnsemble(num_units=3, model_name="openai:gpt-4o", temperature=1.0)

# Create a judge to synthesize the outputs
judge = JudgeSynthesis(model_name="anthropic:claude-3-opus")

# Combine them sequentially
pipeline = Sequential(operators=[ensemble, judge])

# This can now be executed with:
# result = pipeline(inputs={"query": "What is the future of AI?"})
