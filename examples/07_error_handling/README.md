# Error Handling

Build robust applications that gracefully handle failures.

## Overview

These examples demonstrate error handling patterns:
- Catching and recovering from API errors
- Retry strategies with backoff
- Graceful degradation patterns

## Examples

1. **robust_patterns.py** - Production-Ready Error Handling
   - Handle rate limits and timeouts
   - Implement retry with exponential backoff
   - Fall back to alternative models
   - Log errors for debugging

## Key Patterns

### Retry with Backoff

```python
from ember.api import models
import time

def call_with_retry(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            return models("gpt-4o-mini", prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt  # Exponential backoff
            time.sleep(wait)
```

### Model Fallback

```python
FALLBACK_MODELS = ["gpt-4o", "gpt-4o-mini", "claude-3-5-haiku-latest"]

def call_with_fallback(prompt: str) -> str:
    for model_id in FALLBACK_MODELS:
        try:
            return models(model_id, prompt)
        except Exception:
            continue
    raise RuntimeError("All models failed")
```

## Error Categories

| Error Type | Cause | Strategy |
|------------|-------|----------|
| Rate limit | Too many requests | Backoff and retry |
| Timeout | Slow response | Retry or use faster model |
| API key | Missing credentials | Check configuration |
| Model unavailable | Service outage | Fall back to alternative |

## Prerequisites

Requires configured model providers. Examples demonstrate both success
and failure cases.

## Next Steps

- **08_advanced_patterns/** - XCS integration for complex workflows
- **10_evaluation_suite/** - Test and validate your error handling
