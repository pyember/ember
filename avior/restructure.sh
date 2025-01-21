#!/usr/bin/env bash
set -e

echo "Creating required directories..."
mkdir -p src/avior/conf
mkdir -p src/avior/core
mkdir -p src/avior/modules
mkdir -p src/avior/operators
mkdir -p src/avior/registry
mkdir -p src/avior/registry/dataset
mkdir -p src/avior/registry/eval_function
mkdir -p src/avior/registry/models
mkdir -p src/avior/registry/persona
mkdir -p src/avior/registry/prompt
mkdir -p src/avior/tasks
mkdir -p src/avior/tests
mkdir -p src/avior/utils

# -----------------------------------------------------------------------------
# Core files
# -----------------------------------------------------------------------------
# Move existing core files (graph_executor.py, module.py, simple_flow.py, etc.) to src/avior/core/
# If they already live there, this mv may fail or do nothing, which is okay:
mv src/avior/core/graph_executor.py src/avior/core/ 2>/dev/null || true
mv src/avior/core/module.py         src/avior/core/ 2>/dev/null || true
mv src/avior/core/simple_flow.py    src/avior/core/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
mv src/avior/modules/lm_modules.py src/avior/modules/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------
mv src/avior/operators/operator_base.py    src/avior/operators/ 2>/dev/null || true
mv src/avior/operators/operator_registry.py src/avior/operators/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
# Keep model_registry.py where it belongs in src/avior/registry/models:
mv src/avior/registry/models/model_registry.py src/avior/registry/models/ 2>/dev/null || true
mv src/avior/registry/models/models.py         src/avior/registry/models/ 2>/dev/null || true
mv src/avior/registry/models/domain_models.py  src/avior/registry/models/ 2>/dev/null || true
mv src/avior/registry/models/factory.py        src/avior/registry/models/ 2>/dev/null || true
mv src/avior/registry/models/usage_tracker.py  src/avior/registry/models/ 2>/dev/null || true

# Keep dataset_registry.py in src/avior/registry/dataset:
mv src/avior/registry/dataset/dataset_registry.py src/avior/registry/dataset/ 2>/dev/null || true

# Keep eval_function_registry.py in src/avior/registry/eval_function:
mv src/avior/registry/eval_function/eval_function_registry.py src/avior/registry/eval_function/ 2>/dev/null || true

# Move persona registry:
mv src/avior/registry/persona/persona_registry.py src/avior/registry/persona/ 2>/dev/null || true

# Move prompt registry (and keep signatures.py if you want to retain it):
mv src/avior/registry/prompt/prompt_registry.py src/avior/registry/prompt/ 2>/dev/null || true
mv src/avior/registry/prompt/signatures.py      src/avior/registry/prompt/ 2>/dev/null || true

# Move non.py:
mv src/avior/registry/non.py src/avior/registry/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
mv src/avior/utils/dataset_utils.py  src/avior/utils/ 2>/dev/null || true
mv src/avior/utils/multeity_utils.py src/avior/utils/ 2>/dev/null || true
mv src/avior/utils/retry_utils.py    src/avior/utils/ 2>/dev/null || true
mv src/avior/utils/usage_utils.py    src/avior/utils/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
# The “tests/avior/…” folders match your new structure:
# If you see duplicates (e.g. test_model_registry.py in two places), move or rename one as needed.
mkdir -p tests/avior
mkdir -p tests/avior/core
mkdir -p tests/avior/modules
mkdir -p tests/avior/operators
mkdir -p tests/avior/registry
mkdir -p tests/avior/registry/dataset
mkdir -p tests/avior/registry/eval_function
mkdir -p tests/avior/registry/models
mkdir -p tests/avior/registry/prompt
mkdir -p tests/avior/tasks
mkdir -p tests/avior/utils

# Move test files. Where these already match the new layout, you can skip or ignore errors:
mv tests/avior/core/test_avior_graph.py  tests/avior/core/ 2>/dev/null || true
mv tests/avior/core/test_graph_executor.py tests/avior/core/ 2>/dev/null || true
mv tests/avior/core/test_non_graph.py     tests/avior/core/ 2>/dev/null || true
mv tests/avior/modules/test_lm_modules.py tests/avior/modules/ 2>/dev/null || true
mv tests/avior/operators/test_operator.py tests/avior/operators/ 2>/dev/null || true
mv tests/avior/operators/test_operators.py tests/avior/operators/ 2>/dev/null || true
mv tests/avior/registry/test_integration.py tests/avior/registry/ 2>/dev/null || true
mv tests/avior/registry/test_non.py         tests/avior/registry/ 2>/dev/null || true
mv tests/avior/registry/models/test_model_registry.py tests/avior/registry/models/ 2>/dev/null || true
mv tests/avior/registry/models/test_domain_models.py tests/avior/registry/models/ 2>/dev/null || true
mv tests/avior/registry/models/test_factory.py       tests/avior/registry/models/ 2>/dev/null || true
mv tests/avior/registry/models/test_models.py        tests/avior/registry/models/ 2>/dev/null || true
mv tests/avior/registry/models/test_usage_tracker.py tests/avior/registry/models/ 2>/dev/null || true
mv tests/avior/registry/dataset/test_dataset_registry.py tests/avior/registry/dataset/ 2>/dev/null || true
mv tests/avior/registry/eval_function/test_eval_function_registry.py tests/avior/registry/eval_function/ 2>/dev/null || true
mv tests/avior/registry/prompt/__init__.py  tests/avior/registry/prompt/ 2>/dev/null || true
mv tests/avior/tasks/test_some_task.py tests/avior/tasks/ 2>/dev/null || true
mv tests/avior/utils/test_dataset_utils.py tests/avior/utils/ 2>/dev/null || true

# If you have any second copy of “test_model_registry.py” at tests/avior/registry/test_model_registry.py
# and do not want to delete it, rename or move it:
mv tests/avior/registry/test_model_registry.py tests/avior/registry/models/test_model_registry_extra.py 2>/dev/null || true

echo "Restructure complete!"