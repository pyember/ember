#!/usr/bin/env bash
set -e

echo "Creating required directories..."
mkdir -p src/ember/conf
mkdir -p src/ember/core
mkdir -p src/ember/modules
mkdir -p src/ember/operators
mkdir -p src/ember/registry
mkdir -p src/ember/registry/dataset
mkdir -p src/ember/registry/eval_function
mkdir -p src/ember/registry/models
mkdir -p src/ember/registry/persona
mkdir -p src/ember/registry/prompt
mkdir -p src/ember/tasks
mkdir -p src/ember/tests
mkdir -p src/ember/utils

# -----------------------------------------------------------------------------
# Core files
# -----------------------------------------------------------------------------
# Move existing core files (graph_executor.py, module.py, simple_flow.py, etc.) to src/ember/core/
# If they already live there, this mv may fail or do nothing, which is okay:
mv src/ember/core/graph_executor.py src/ember/core/ 2>/dev/null || true
mv src/ember/core/module.py         src/ember/core/ 2>/dev/null || true
mv src/ember/core/simple_flow.py    src/ember/core/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
mv src/ember/modules/lm_modules.py src/ember/modules/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------
mv src/ember/operators/operator_base.py    src/ember/operators/ 2>/dev/null || true
mv src/ember/operators/operator_registry.py src/ember/operators/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
# Keep model_registry.py where it belongs in src/ember/registry/models:
mv src/ember/registry/models/model_registry.py src/ember/registry/models/ 2>/dev/null || true
mv src/ember/registry/models/models.py         src/ember/registry/models/ 2>/dev/null || true
mv src/ember/registry/models/domain_models.py  src/ember/registry/models/ 2>/dev/null || true
mv src/ember/registry/models/factory.py        src/ember/registry/models/ 2>/dev/null || true
mv src/ember/registry/models/usage_tracker.py  src/ember/registry/models/ 2>/dev/null || true

# Keep dataset_registry.py in src/ember/registry/dataset:
mv src/ember/registry/dataset/dataset_registry.py src/ember/registry/dataset/ 2>/dev/null || true

# Keep eval_function_registry.py in src/ember/registry/eval_function:
mv src/ember/registry/eval_function/eval_function_registry.py src/ember/registry/eval_function/ 2>/dev/null || true

# Move persona registry:
mv src/ember/registry/persona/persona_registry.py src/ember/registry/persona/ 2>/dev/null || true

# Move prompt registry (and keep signatures.py if you want to retain it):
mv src/ember/registry/prompt/prompt_registry.py src/ember/registry/prompt/ 2>/dev/null || true
mv src/ember/registry/prompt/signatures.py      src/ember/registry/prompt/ 2>/dev/null || true

# Move non.py:
mv src/ember/registry/non.py src/ember/registry/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
mv src/ember/utils/dataset_utils.py  src/ember/utils/ 2>/dev/null || true
mv src/ember/utils/multeity_utils.py src/ember/utils/ 2>/dev/null || true
mv src/ember/utils/retry_utils.py    src/ember/utils/ 2>/dev/null || true
mv src/ember/utils/usage_utils.py    src/ember/utils/ 2>/dev/null || true

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
# The “tests/ember/…” folders match your new structure:
# If you see duplicates (e.g. test_model_registry.py in two places), move or rename one as needed.
mkdir -p tests/ember
mkdir -p tests/ember/core
mkdir -p tests/ember/modules
mkdir -p tests/ember/operators
mkdir -p tests/ember/registry
mkdir -p tests/ember/registry/dataset
mkdir -p tests/ember/registry/eval_function
mkdir -p tests/ember/registry/models
mkdir -p tests/ember/registry/prompt
mkdir -p tests/ember/tasks
mkdir -p tests/ember/utils

# Move test files. Where these already match the new layout, you can skip or ignore errors:
mv tests/ember/core/test_ember_graph.py  tests/ember/core/ 2>/dev/null || true
mv tests/ember/core/test_graph_executor.py tests/ember/core/ 2>/dev/null || true
mv tests/ember/core/test_non_graph.py     tests/ember/core/ 2>/dev/null || true
mv tests/ember/modules/test_lm_modules.py tests/ember/modules/ 2>/dev/null || true
mv tests/ember/operators/test_operator.py tests/ember/operators/ 2>/dev/null || true
mv tests/ember/operators/test_operators.py tests/ember/operators/ 2>/dev/null || true
mv tests/ember/registry/test_integration.py tests/ember/registry/ 2>/dev/null || true
mv tests/ember/registry/test_non.py         tests/ember/registry/ 2>/dev/null || true
mv tests/ember/registry/models/test_model_registry.py tests/ember/registry/models/ 2>/dev/null || true
mv tests/ember/registry/models/test_domain_models.py tests/ember/registry/models/ 2>/dev/null || true
mv tests/ember/registry/models/test_factory.py       tests/ember/registry/models/ 2>/dev/null || true
mv tests/ember/registry/models/test_models.py        tests/ember/registry/models/ 2>/dev/null || true
mv tests/ember/registry/models/test_usage_tracker.py tests/ember/registry/models/ 2>/dev/null || true
mv tests/ember/registry/dataset/test_dataset_registry.py tests/ember/registry/dataset/ 2>/dev/null || true
mv tests/ember/registry/eval_function/test_eval_function_registry.py tests/ember/registry/eval_function/ 2>/dev/null || true
mv tests/ember/registry/prompt/__init__.py  tests/ember/registry/prompt/ 2>/dev/null || true
mv tests/ember/tasks/test_some_task.py tests/ember/tasks/ 2>/dev/null || true
mv tests/ember/utils/test_dataset_utils.py tests/ember/utils/ 2>/dev/null || true

# If you have any second copy of “test_model_registry.py” at tests/ember/registry/test_model_registry.py
# and do not want to delete it, rename or move it:
mv tests/ember/registry/test_model_registry.py tests/ember/registry/models/test_model_registry_extra.py 2>/dev/null || true

echo "Restructure complete!"