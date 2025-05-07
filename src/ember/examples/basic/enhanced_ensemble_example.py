"""Enhanced Ensemble Example

This module demonstrates how to use an ensemble of PromptEnhancerOperators
with a JudgeSynthesisOperator to produce a high-quality enhanced query.

The example creates multiple PromptEnhancerOperator instances in an ensemble,
then uses JudgeSynthesisOperator to select the best enhanced query from all
the ensemble responses.

To run:
    uv run python src/ember/examples/basic/enhanced_ensemble_example.py
"""
from ember.core import non

system = non.Sequential(
    operators=[
        non.PromptEnhancer(model_name="gpt-4o-mini"),
        non.UniformEnsemble(num_units=4, model_name="gpt-4.1", temperature=1),
        non.JudgeSynthesis(model_name="gpt-4.1", temperature=0.0),
    ]
)

result = system(inputs={"query": "What causes climate change?"})

print(result.final_answer)
