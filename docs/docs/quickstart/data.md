# Ember Data Processing - Quickstart Guide

This guide introduces Ember's data processing system, which provides tools for loading, transforming, and evaluating data across various benchmarks and datasets.

## 1. Introduction to Ember Data

Ember's data module provides:
- **Standardized Dataset Access**: Unified interface to common benchmarks and custom datasets
- **Flexible Transformations**: Pipelines for preprocessing and normalizing data
- **Evaluation Framework**: Tools for measuring model performance across tasks
- **Sampling Controls**: Methods for dataset subsampling and stratification
- **Data Registry**: Central registry of popular evaluation benchmarks with metadata

## 2. Loading Datasets - The Simple Way

```python
from ember.api import datasets

# Load a standard benchmark dataset
mmlu_data = datasets("mmlu", config={"subset": "high_school_biology", "split": "test"})

# Access dataset entries
for entry in mmlu_data:
    print(f"Question: {entry.query}")
    print(f"Choices: {entry.choices}")
    print(f"Answer: {entry.metadata.get('correct_answer')}")
    print("---")
```

## 3. Using DatasetBuilder

```python
from ember.api import DatasetBuilder

# Create and configure dataset using builder pattern
transformed_data = (DatasetBuilder()
    .from_registry("mmlu")
    .subset("high_school_biology")
    .split("test")
    .sample(100)
    .transform(lambda item: {
        **item,
        "question": f"Please answer: {item['question']}"
    })
    .build())

# Access the transformed data
for entry in transformed_data:
    print(f"Question: {entry.query}")
    print(f"Choices: {entry.choices}")
    print(f"Answer: {entry.metadata.get('correct_answer')}")
    print("---")
```

## 4. Creating Custom Datasets

```python
from ember.api import register, Dataset, DatasetEntry, TaskType
from typing import List, Dict, Any
import json

# Define dataset class with registration decorator
@register("my_dataset", source="custom/qa", task_type=TaskType.QUESTION_ANSWERING)
class CustomDataset:
    def load(self, config=None) -> List[DatasetEntry]:
        # Load data from a file
        with open("my_dataset.json", "r") as f:
            data = json.load(f)
        
        # Convert to DatasetEntry objects
        entries = []
        for entry in data:
            item = DatasetEntry(
                id=entry["id"],
                content={
                    "question": entry["question"],
                    "choices": entry["options"],
                    "answer": entry["correct_option"]
                },
                metadata={
                    "category": entry["category"],
                    "difficulty": entry["difficulty"]
                }
            )
            entries.append(item)
        
        return entries

# Use it like any other dataset
my_data = datasets("my_dataset")
```

## 5. Filtering and Transforming Datasets

```python
from ember.api import DatasetBuilder

# Custom transformation function
def add_context_to_question(item):
    return {
        **item,
        "question": f"In the context of {item['metadata']['category']}: {item['question']}"
    }

# Filtering to specific categories
science_questions = (DatasetBuilder()
    .from_registry("mmlu")
    .subset("high_school_chemistry")
    .filter(lambda item: "reaction" in item["question"].lower())
    .transform(add_context_to_question)
    .build())

print(f"Found {len(science_questions)} chemistry reaction questions")
```

## 6. Evaluating Model Performance

```python
from ember.api import datasets
from ember.core.utils.eval.pipeline import EvaluationPipeline
from ember.core.utils.eval.evaluators import MultipleChoiceEvaluator
from ember.api.models import ModelBuilder

# Load a dataset
mmlu_data = datasets("mmlu", config={"subset": "high_school_biology", "split": "test"})

# Initialize model
model = ModelBuilder().temperature(0.0).build("openai:gpt-4o")

# Create evaluator
evaluator = MultipleChoiceEvaluator()

# Set up and run evaluation pipeline
eval_pipeline = EvaluationPipeline(
    dataset=mmlu_data.entries,
    evaluators=[evaluator],
    model=model
)

# Run evaluation
results = eval_pipeline.evaluate()

# Print results
print(f"Accuracy: {results.metrics['accuracy']:.2f}")
print(f"Per-category breakdown: {results.metrics.get('category_accuracy', {})}")
```

## 7. Working with Evaluation Results

```python
import matplotlib.pyplot as plt

# Assuming we have evaluation results
# results: EvaluationResults

# Access overall metrics
accuracy = results.metrics["accuracy"]
f1 = results.metrics.get("f1_score", 0.0)

# Access per-item results
for item_result in results.item_results:
    item_id = item_result.item_id
    correct = item_result.correct
    model_answer = item_result.model_output
    expected = item_result.expected_output
    
    if not correct:
        print(f"Item {item_id} was incorrect:")
        print(f"  Expected: {expected}")
        print(f"  Model output: {model_answer}")

# Plot results if category data is available
if "category_accuracy" in results.metrics:
    categories = results.metrics["category_accuracy"].keys()
    accuracies = [results.metrics["category_accuracy"][cat] for cat in categories]
    
    plt.figure(figsize=(10, 6))
    plt.bar(categories, accuracies)
    plt.title("Accuracy by Category")
    plt.xlabel("Category")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("category_results.png")
```

## 8. Built-in Datasets

Ember provides ready-to-use loaders for popular benchmarks:

```python
from ember.api import datasets, list_available_datasets

# List all available datasets
available_datasets = list_available_datasets()
print(f"Available datasets: {available_datasets}")

# MMLU (Massive Multitask Language Understanding)
mmlu = datasets("mmlu", config={"subset": "high_school_mathematics"})

# TruthfulQA
truthful_qa = datasets("truthful_qa", config={"subset": "generation"})

# HaluEval (Hallucination Evaluation)
halu_eval = datasets("halueval", config={"subset": "knowledge"})

# CommonsenseQA
commonsense_qa = datasets("commonsense_qa", config={"split": "validation"})

# AIME (American Invitational Mathematics Examination)
aime = datasets("aime")

# GPQA (Graduate-level Physics Questions)
gpqa = datasets("gpqa")

# Codeforces Programming Problems
codeforces = datasets("codeforces", config={"difficulty_range": (800, 1200)})
```

## 9. Best Practices

1. **Use the Builder Pattern**: `DatasetBuilder` provides a clean, fluent interface
2. **Registry Integration**: Register custom datasets for seamless integration
3. **Transformation Order**: Consider the sequence of transformations and filters
4. **Stratified Sampling**: Ensure representative subsets with appropriate sampling
5. **Multiple Evaluators**: Use specialized evaluators for comprehensive assessment
6. **Configuration**: Use structured configs for reproducibility

## Next Steps

Learn more about:
- [Model Registry](model_registry.md) - Managing LLM configurations
- [Operators](operators.md) - Building computational units
- [Evaluation Metrics](../advanced/evaluation_metrics.md) - Detailed metrics and evaluation approaches
- [Custom Transformers](../advanced/custom_transformers.md) - Building custom data transformations