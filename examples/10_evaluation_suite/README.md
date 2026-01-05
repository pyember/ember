# Evaluation Suite

Test and benchmark your Ember applications.

## Overview

These examples demonstrate evaluation and testing patterns:
- Accuracy measurement on benchmarks
- Consistency testing across runs
- Building evaluation harnesses

## Examples

1. **accuracy_evaluation.py** - Measure Model Accuracy
   - Evaluate on standard benchmarks
   - Compute accuracy metrics
   - Compare model performance

2. **benchmark_harness.py** - Build Evaluation Pipelines
   - Create reusable evaluation frameworks
   - Run systematic experiments
   - Generate performance reports

3. **consistency_testing.py** - Test Response Consistency
   - Verify reproducibility
   - Detect model drift
   - Validate temperature effects

## Key APIs

```python
from ember.api import load, stream
from ember.api.eval import Evaluator, EvaluationPipeline

# Load evaluation dataset
dataset = load("mmlu", max_items=100)

# Create evaluator
evaluator = Evaluator(
    metric="accuracy",
    extractor="multiple_choice"
)

# Run evaluation
results = []
for record in dataset:
    prediction = models("gpt-4o-mini", record.question)
    score = evaluator.evaluate(prediction, record.answer)
    results.append(score)

accuracy = sum(results) / len(results)
```

## Evaluation Patterns

### A/B Model Comparison

```python
def compare_models(dataset, model_a: str, model_b: str) -> dict:
    scores = {"model_a": [], "model_b": []}

    for record in dataset:
        pred_a = models(model_a, record.question)
        pred_b = models(model_b, record.question)

        scores["model_a"].append(evaluate(pred_a, record.answer))
        scores["model_b"].append(evaluate(pred_b, record.answer))

    return {
        "model_a_accuracy": sum(scores["model_a"]) / len(scores["model_a"]),
        "model_b_accuracy": sum(scores["model_b"]) / len(scores["model_b"]),
    }
```

### Consistency Check

```python
def check_consistency(prompt: str, n_runs: int = 5) -> float:
    responses = [models("gpt-4o-mini", prompt, temperature=0) for _ in range(n_runs)]
    unique = len(set(responses))
    return 1.0 - (unique - 1) / n_runs  # 1.0 = perfectly consistent
```

## Prerequisites

- Configured model providers for model evaluation
- Dataset access (most built-in datasets work offline)

## Metrics Reference

| Metric | Use Case |
|--------|----------|
| Accuracy | Classification, multiple choice |
| F1 Score | Imbalanced datasets |
| BLEU/ROUGE | Text generation quality |
| Exact match | Factual QA |

## Next Steps

You've completed the examples. Consider:
- Building your own evaluation harness
- Contributing new patterns back to Ember
- Exploring the API reference documentation
