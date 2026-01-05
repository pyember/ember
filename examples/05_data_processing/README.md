# Data Processing

Work with datasets and data streams in Ember.

## Overview

These examples demonstrate Ember's data handling capabilities:
- Loading datasets from various sources
- Streaming data for memory-efficient processing
- Integrating data with model calls

## Examples

1. **loading_datasets.py** - Load and Query Datasets
   - Load built-in benchmark datasets
   - Access dataset metadata
   - Query and filter records

2. **streaming_data.py** - Stream Large Datasets
   - Process data without loading everything into memory
   - Handle large-scale evaluations
   - Combine streaming with model inference

## Key APIs

```python
from ember.api import load, stream, list_datasets, metadata

# Discover available datasets
datasets = list_datasets()

# Load dataset into memory
data = load("mmlu", max_items=100)

# Stream for large datasets
for record in stream("mmlu", max_items=1000):
    process(record)

# Get dataset info
info = metadata("mmlu")
```

## Prerequisites

Most datasets require no API keys to load. Model inference examples require
configured providers.

## Next Steps

After understanding data handling, explore:
- **06_performance_optimization/** - Batch processing and JIT compilation
- **09_practical_patterns/** - RAG and structured output patterns
