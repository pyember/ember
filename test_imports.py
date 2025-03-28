"""Test script to verify our import fix."""

print("Testing imports...")

# Import from the API module
from ember.api import DatasetBuilder, Evaluator, EvaluationPipeline

print("✅ Successfully imported DatasetBuilder from ember.api")
print("✅ Successfully imported Evaluator from ember.api")
print("✅ Successfully imported EvaluationPipeline from ember.api")

# Import specifically from data module
from ember.api.data import DatasetBuilder, Dataset

print("✅ Successfully imported DatasetBuilder from ember.api.data")

# Import specifically from eval module
from ember.api.eval import Evaluator, EvaluationPipeline

print("✅ Successfully imported Evaluator and EvaluationPipeline from ember.api.eval")

print("All imports successful!")
