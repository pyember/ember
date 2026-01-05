"""Facade exposing Ember's stable application programming interface.

The :mod:`ember.api` package consolidates high-level helpers for model
invocation, operator composition, dataset access, evaluation, and validation.
All public entry points are curated here to provide a single import surface for
applications and notebooks.

Examples:
    >>> from ember.api import models
    >>> summary = models("gpt-4", "Summarize 2+2")
    >>> summary
    '4'

    >>> from ember.api.operators import op
    >>> @op
    ... def summarize(text: str) -> str:
    ...     return models("gpt-4", f"Summarize: {text}")
"""

# Module imports
import ember.api.eval as evaluation
import ember.api.exceptions as exceptions
import ember.api.types as types
import ember.api.validators as validators
import ember.api.xcs as xcs
import ember.non as non

# Specific imports
from ember.api.data import (
    DatasetInfo,
    DataSource,
    FileSource,
    HuggingFaceSource,
    StreamIterator,
    from_file,
    list_datasets,
    load,
    load_file,
    metadata,
    register,
    stream,
)
from ember.api.decorators import (
    mark_hybrid,
    mark_orchestration,
    mark_pytree_safe,
    mark_tensor,
    op,
)
from ember.api.eval import EvaluationPipeline, Evaluator
from ember.api.media_cache import DiskMediaCache
from ember.api.models import models
from ember.api.record import (
    Choice,
    ChoiceSet,
    DataRecord,
    DatasetRef,
    MediaAsset,
    MediaBundle,
    TextContent,
)
from ember.api.validators import (
    ValidationHelpers,
    field_validator,
    model_validator,
)
from ember.models.catalog import Models

__all__ = [
    # Core facades
    "models",
    "Models",
    "op",
    # XCS classification markers
    "mark_orchestration",
    "mark_tensor",
    "mark_hybrid",
    "mark_pytree_safe",
    # Module namespaces (operators excluded - import from ember.api.operators)
    "evaluation",
    "types",
    "validators",
    "xcs",
    "non",
    "exceptions",
    # Data API
    "stream",
    "load",
    "metadata",
    "list_datasets",
    "register",
    "from_file",
    "load_file",
    "DataSource",
    "DatasetInfo",
    "StreamIterator",
    "FileSource",
    "HuggingFaceSource",
    "DiskMediaCache",
    # Data record abstractions
    "DataRecord",
    "TextContent",
    "Choice",
    "ChoiceSet",
    "MediaAsset",
    "MediaBundle",
    "DatasetRef",
    # Evaluation API
    "Evaluator",
    "EvaluationPipeline",
    # Validation API
    "field_validator",
    "model_validator",
    "ValidationHelpers",
]
