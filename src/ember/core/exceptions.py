"""
Exception architecture for the Ember framework.

This module defines a hierarchical, domain-driven exception system that provides
consistent error reporting, contextual details, and actionable messages across
all Ember components.

All exceptions follow these design principles:
1. Rich context: Exceptions include detailed context for debugging
2. Domain specificity: Exception hierarchy mirrors the domain structure
3. Actionable messages: Error messages suggest potential fixes
4. Consistent formatting: Standard approach to error representation
5. Unified error codes: Centralized management of error codes
"""

import inspect
import logging
from typing import Any, ClassVar, Dict, List, Optional, TypedDict

# Error Code Ranges (reserved spaces for different modules)
# 1000-1999: Core Framework
# 2000-2999: Operator Framework
# 3000-3999: Model Framework
# 4000-4999: Data Framework
# 5000-5999: XCS Framework
# 6000-6999: Configuration
# 7000-7999: API
# 8000-8999: CLI
# 9000-9999: Plugin System


class ExceptionContext(TypedDict, total=False):
    """Standard type for exception context with common fields."""

    caller_file: str
    caller_function: str
    caller_lineno: int
    cause_type: str
    cause_message: str


class EmberError(Exception):
    """Base class for all Ember exceptions with enhanced context and error codes.

    All Ember exceptions inherit from this class, providing a consistent interface
    for error handling, reporting, and contextual information to aid debugging
    and provide actionable guidance.

    Attributes:
        message: The error message
        error_code: Unique error code for this exception type
        context: Additional context information as key-value pairs
        recovery_hint: Optional hint for how to recover from this error
    """

    # Class-level constants
    DEFAULT_ERROR_CODE: ClassVar[Optional[int]] = None
    DEFAULT_RECOVERY_HINT: ClassVar[Optional[str]] = None

    # Instance attributes
    message: str
    error_code: Optional[int]
    context: Dict[str, Any]
    recovery_hint: Optional[str]

    def __init__(
        self,
        message: str,
        error_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        recovery_hint: Optional[str] = None,
    ) -> None:
        """Initialize with message, error code, and optional context.

        Args:
            message: The error message
            error_code: Optional error code (defaults to class DEFAULT_ERROR_CODE)
            context: Optional dictionary of contextual information
            cause: Optional exception that caused this error
            recovery_hint: Optional hint for how to recover from this error
        """
        self.message = message
        self.error_code = (
            error_code if error_code is not None else self.DEFAULT_ERROR_CODE
        )
        self.context = context or {}
        self.recovery_hint = recovery_hint or self.DEFAULT_RECOVERY_HINT
        self.__cause__ = cause

        if cause:
            self.add_context(cause_type=type(cause).__name__, cause_message=str(cause))

        # Add caller information to the context for better traceability
        if (
            self.__class__.DEFAULT_ERROR_CODE is not None
        ):  # Only add for concrete exceptions
            frame = inspect.currentframe()
            if frame:
                frame = frame.f_back  # Get the caller's frame
                if frame:
                    caller_info = inspect.getframeinfo(frame)
                    self.add_context(
                        caller_file=caller_info.filename,
                        caller_function=caller_info.function,
                        caller_lineno=caller_info.lineno,
                    )

        super().__init__(self._format_message())

    def add_context(self, **kwargs: Any) -> "EmberError":
        """Add additional context information to the exception.

        Args:
            **kwargs: Key-value pairs to add to the context

        Returns:
            Self for method chaining
        """
        self.context.update(kwargs)
        return self

    def get_context(self) -> Dict[str, Any]:
        """Get a copy of the context dictionary.

        Returns:
            Dictionary containing contextual information
        """
        return self.context.copy()

    def _format_message(self) -> str:
        """Format the error message with error code, context, and recovery hint.

        Returns:
            Formatted error message string
        """
        parts = []
        if self.error_code is not None:
            parts.append(f"[Error {self.error_code}]")

        parts.append(self.message)

        if self.recovery_hint:
            parts.append(f"[Recovery: {self.recovery_hint}]")

        if self.context:
            # Format context as key=value pairs
            context_str = ", ".join(
                f"{k}={v!r}" for k, v in sorted(self.context.items())
            )
            parts.append(f"[Context: {context_str}]")

        return " ".join(parts)

    def log_with_context(
        self, logger: logging.Logger, level: int = logging.ERROR
    ) -> None:
        """Log the error with its full context.

        Args:
            logger: Logger to use for recording the error
            level: Logging level (default: ERROR)
        """
        logger.log(
            level,
            f"{self.__class__.__name__}: {self.message}",
            extra={"structured_data": self.get_context()},
        )

    @classmethod
    def from_exception(
        cls, exception: Exception, message: Optional[str] = None, **context: Any
    ) -> "EmberError":
        """Create an EmberError from another exception.

        This factory method wraps any exception in an EmberError, preserving
        the original exception as the cause.

        Args:
            exception: The exception to wrap
            message: Optional message (defaults to exception's message)
            **context: Additional context to include

        Returns:
            A new EmberError instance
        """
        if message is None:
            message = str(exception)

        return cls(message=message, cause=exception, context=context)

    @classmethod
    def with_context(cls, message: str, **context: Any) -> "EmberError":
        """Create an exception with context.

        This factory method creates a new exception with the given message and context.

        Args:
            message: The error message
            **context: Context fields to include

        Returns:
            A new exception instance with the specified context
        """
        return cls(message=message, context=context)


class ErrorGroup(EmberError):
    """Container for multiple related errors.

    This exception type allows aggregating multiple related errors into a single
    exception, making it easier to report and handle multiple issues together.

    Attributes:
        errors: List of exceptions contained in this group
    """

    DEFAULT_ERROR_CODE = 1000

    def __init__(
        self,
        message: str,
        errors: List[Exception],
        error_code: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize with a message and list of errors.

        Args:
            message: The error message
            errors: List of exceptions to include in this group
            error_code: Optional error code
            context: Optional context dictionary
        """
        self.errors = errors

        # Add error count to context
        error_context = {} if context is None else dict(context)
        error_context["error_count"] = len(errors)

        super().__init__(
            message=message,
            error_code=error_code,
            context=error_context,
        )

    def _format_message(self) -> str:
        """Format the error message with the list of contained errors.

        Returns:
            Formatted error message string
        """
        base_message = super()._format_message()
        error_details = "\n".join(
            f"  - {e.__class__.__name__}: {str(e)}" for e in self.errors
        )
        return f"{base_message}\nContained errors:\n{error_details}"


# =========================================================
# Core Framework Exceptions (1000-1999)
# =========================================================


class InvalidArgumentError(EmberError):
    """Raised when an invalid argument is provided to a function or method."""

    DEFAULT_ERROR_CODE = 1001
    DEFAULT_RECOVERY_HINT = "Check the function signature and provide valid arguments"


class ValidationError(EmberError):
    """Raised when input validation fails."""

    DEFAULT_ERROR_CODE = 1002
    DEFAULT_RECOVERY_HINT = "Ensure the input data meets all validation requirements"


class NotImplementedFeatureError(EmberError):
    """Raised when a required feature is not yet implemented."""

    DEFAULT_ERROR_CODE = 1003
    DEFAULT_RECOVERY_HINT = "Check the documentation for supported features"


class DeprecationError(EmberError):
    """Raised when a deprecated feature is used."""

    DEFAULT_ERROR_CODE = 1004
    DEFAULT_RECOVERY_HINT = "Update your code to use the recommended alternative"


class IncompatibleTypeError(EmberError):
    """Raised when incompatible types are used together."""

    DEFAULT_ERROR_CODE = 1005
    DEFAULT_RECOVERY_HINT = "Ensure you're using compatible types"


class InitializationError(EmberError):
    """Raised during object initialization failures."""

    DEFAULT_ERROR_CODE = 1006
    DEFAULT_RECOVERY_HINT = "Check the initialization parameters and requirements"


# =========================================================
# Configuration Exceptions (6000-6999)
# =========================================================


class ConfigError(EmberError):
    """Base class for configuration-related errors."""

    DEFAULT_ERROR_CODE = 6000
    DEFAULT_RECOVERY_HINT = "Check your configuration settings"


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    DEFAULT_ERROR_CODE = 6001
    DEFAULT_RECOVERY_HINT = "Ensure your configuration values meet the requirements"


class ConfigFileError(ConfigError):
    """Raised when there's an issue with a configuration file."""

    DEFAULT_ERROR_CODE = 6002
    DEFAULT_RECOVERY_HINT = (
        "Check that the configuration file exists and has the correct format"
    )


class ConfigValueError(ConfigError):
    """Raised when a configuration value is invalid."""

    DEFAULT_ERROR_CODE = 6003
    DEFAULT_RECOVERY_HINT = "Check the allowed values for this configuration setting"


class MissingConfigError(ConfigError):
    """Raised when a required configuration is missing."""

    DEFAULT_ERROR_CODE = 6004
    DEFAULT_RECOVERY_HINT = "Add the missing configuration key to your config file"


# =========================================================
# Registry Exceptions (1100-1199)
# =========================================================


class RegistryError(EmberError):
    """Base class for registry-related errors."""

    DEFAULT_ERROR_CODE = 1100
    DEFAULT_RECOVERY_HINT = "Check your registry configuration and operations"


class ItemNotFoundError(RegistryError):
    """Raised when an item is not found in a registry."""

    DEFAULT_ERROR_CODE = 1101
    DEFAULT_RECOVERY_HINT = "Verify the item exists and is correctly registered"

    @classmethod
    def for_item(
        cls, item_name: str, registry_name: Optional[str] = None
    ) -> "ItemNotFoundError":
        """Create an exception for a specific item not found.

        Args:
            item_name: Name of the item that wasn't found
            registry_name: Optional name of the registry

        Returns:
            A new ItemNotFoundError
        """
        registry_text = f" in {registry_name}" if registry_name else ""
        message = f"Item '{item_name}' not found{registry_text}"
        context = {"item_name": item_name}

        if registry_name:
            context["registry_name"] = registry_name

        return cls(message=message, context=context)


class DuplicateItemError(RegistryError):
    """Raised when attempting to register a duplicate item."""

    DEFAULT_ERROR_CODE = 1102
    DEFAULT_RECOVERY_HINT = "Use a unique name or key for the item"


class RegistrationError(RegistryError):
    """Raised when registration of an item fails."""

    DEFAULT_ERROR_CODE = 1103
    DEFAULT_RECOVERY_HINT = "Check that the item meets all registration requirements"


# =========================================================
# Model Framework Exceptions (3000-3999)
# =========================================================


class ModelContext(ExceptionContext):
    """Type for model exception context."""

    model_name: str
    provider_name: Optional[str]


class ModelError(EmberError):
    """Base class for model-related errors."""

    DEFAULT_ERROR_CODE = 3000
    DEFAULT_RECOVERY_HINT = "Check your model configuration and usage"


class ModelProviderError(ModelError):
    """Raised when there's an issue with a model provider."""

    DEFAULT_ERROR_CODE = 3001
    DEFAULT_RECOVERY_HINT = "Check provider API access and configuration"

    @classmethod
    def for_provider(
        cls, provider_name: str, message: str, cause: Optional[Exception] = None
    ) -> "ModelProviderError":
        """Create an exception for a specific provider error.

        Args:
            provider_name: Name of the provider
            message: Error message
            cause: Optional exception that caused this error

        Returns:
            A new ModelProviderError
        """
        return cls(
            message=message, context={"provider_name": provider_name}, cause=cause
        )


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""

    DEFAULT_ERROR_CODE = 3002
    DEFAULT_RECOVERY_HINT = "Check the model name and ensure it's correctly registered"

    @classmethod
    def for_model(
        cls, model_name: str, provider_name: Optional[str] = None
    ) -> "ModelNotFoundError":
        """Create an exception for a specific model not found.

        Args:
            model_name: Name of the model that wasn't found
            provider_name: Optional name of the provider

        Returns:
            A new ModelNotFoundError
        """
        provider_text = f" from provider '{provider_name}'" if provider_name else ""
        message = f"Model '{model_name}' not found{provider_text}"
        context: Dict[str, Any] = {"model_name": model_name}

        if provider_name:
            context["provider_name"] = provider_name

        return cls(message=message, context=context)


class ProviderAPIError(ModelError):
    """Raised when an external provider API call fails."""

    DEFAULT_ERROR_CODE = 3010
    DEFAULT_RECOVERY_HINT = "Check API access, credentials, and request parameters"

    @classmethod
    def for_provider(
        cls,
        provider_name: str,
        message: str,
        status_code: Optional[int] = None,
        cause: Optional[Exception] = None,
    ) -> "ProviderAPIError":
        """Create an exception for a specific provider API error.

        Args:
            provider_name: Name of the provider
            message: Error message
            status_code: Optional HTTP status code
            cause: Optional exception that caused this error

        Returns:
            A new ProviderAPIError
        """
        context: Dict[str, Any] = {"provider_name": provider_name}

        if status_code:
            context["status_code"] = status_code

        return cls(message=message, context=context, cause=cause)


class ProviderConfigError(ModelError):
    """Raised when there is an issue with provider configuration."""

    DEFAULT_ERROR_CODE = 3011
    DEFAULT_RECOVERY_HINT = "Check provider credentials and configuration settings"


class ModelDiscoveryError(ModelError):
    """Raised when model discovery fails."""

    DEFAULT_ERROR_CODE = 3020
    DEFAULT_RECOVERY_HINT = "Check provider access and discovery settings"

    @classmethod
    def for_provider(
        cls, provider: str, reason: str, cause: Optional[Exception] = None
    ) -> "ModelDiscoveryError":
        """Create an exception for a specific provider discovery error.

        Args:
            provider: Name of the provider
            reason: Reason for discovery failure
            cause: Optional exception that caused this error

        Returns:
            A new ModelDiscoveryError
        """
        message = f"Discovery failed for provider '{provider}': {reason}"
        return cls(
            message=message,
            context={"provider": provider, "reason": reason},
            cause=cause,
        )


class ModelRegistrationError(ModelError):
    """Raised when model registration fails."""

    DEFAULT_ERROR_CODE = 3021
    DEFAULT_RECOVERY_HINT = "Check model configuration and registry requirements"

    @classmethod
    def for_model(
        cls, model_name: str, reason: str, cause: Optional[Exception] = None
    ) -> "ModelRegistrationError":
        """Create an exception for a specific model registration error.

        Args:
            model_name: Name of the model
            reason: Reason for registration failure
            cause: Optional exception that caused this error

        Returns:
            A new ModelRegistrationError
        """
        message = f"Failed to register model '{model_name}': {reason}"
        return cls(
            message=message,
            context={"model_name": model_name, "reason": reason},
            cause=cause,
        )


class MissingLMModuleError(ModelError):
    """Raised when the expected LM module is missing from an operator."""

    DEFAULT_ERROR_CODE = 3030
    DEFAULT_RECOVERY_HINT = "Ensure the operator has the required LM module configured"


class InvalidPromptError(ModelError):
    """Raised when a prompt is invalid or malformed."""

    DEFAULT_ERROR_CODE = 3040
    DEFAULT_RECOVERY_HINT = "Check prompt format and template variables"


# =========================================================
# Operator Framework Exceptions (2000-2999)
# =========================================================


class OperatorContext(ExceptionContext):
    """Type for operator exception context."""

    operator_name: str
    operator_type: Optional[str]


class OperatorError(EmberError):
    """Base class for operator-related errors."""

    DEFAULT_ERROR_CODE = 2000
    DEFAULT_RECOVERY_HINT = "Check operator configuration and usage"


class OperatorSpecificationError(OperatorError):
    """Raised when there's an issue with an operator specification."""

    DEFAULT_ERROR_CODE = 2001
    DEFAULT_RECOVERY_HINT = "Check operator specification definition"


class OperatorExecutionError(OperatorError):
    """Raised when an error occurs during operator execution."""

    DEFAULT_ERROR_CODE = 2002
    DEFAULT_RECOVERY_HINT = "Check operator inputs and execution environment"

    @classmethod
    def for_operator(
        cls,
        operator_name: str,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context: Any,
    ) -> "OperatorExecutionError":
        """Create an exception for a specific operator execution error.

        Args:
            operator_name: Name of the operator
            message: Optional custom message
            cause: Optional exception that caused this error
            **context: Additional context fields

        Returns:
            A new OperatorExecutionError
        """
        if message is None:
            message = f"Execution error in operator '{operator_name}'"
            if cause:
                message += f": {str(cause)}"

        context_dict = dict(context)
        context_dict["operator_name"] = operator_name

        return cls(message=message, context=context_dict, cause=cause)


class SpecificationValidationError(OperatorError):
    """Raised when input or output specification validation fails."""

    DEFAULT_ERROR_CODE = 2010
    DEFAULT_RECOVERY_HINT = "Ensure inputs match the operator's specification"


class TreeTransformationError(OperatorError):
    """Raised when an error occurs during tree transformation."""

    DEFAULT_ERROR_CODE = 2020
    DEFAULT_RECOVERY_HINT = "Check that tree structure is valid and transformable"


class FlattenError(OperatorError):
    """Raised when flattening an EmberModule fails due to inconsistent field states."""

    DEFAULT_ERROR_CODE = 2030
    DEFAULT_RECOVERY_HINT = (
        "Ensure module fields are in a consistent state before flattening"
    )


# =========================================================
# Data Framework Exceptions (4000-4999)
# =========================================================


class DataContext(ExceptionContext):
    """Type for data exception context."""

    dataset_name: Optional[str]
    field_name: Optional[str]


class DataError(EmberError):
    """Base class for data-related errors."""

    DEFAULT_ERROR_CODE = 4000
    DEFAULT_RECOVERY_HINT = "Check data format and processing operations"


class DataValidationError(DataError):
    """Raised when data validation fails."""

    DEFAULT_ERROR_CODE = 4001
    DEFAULT_RECOVERY_HINT = "Ensure data meets validation requirements"

    @classmethod
    def for_field(
        cls,
        field_name: str,
        message: Optional[str] = None,
        expected_type: Optional[str] = None,
        actual_value: Any = None,
        **additional_context: Any,
    ) -> "DataValidationError":
        """Create an exception for a specific field validation error.

        Args:
            field_name: Name of the field that failed validation
            message: Optional custom message
            expected_type: Optional expected type of the field
            actual_value: Optional actual value received
            **additional_context: Additional context fields to include

        Returns:
            A new DataValidationError
        """
        if message is None:
            message = f"Validation failed for field '{field_name}'"
            if expected_type:
                message += f", expected {expected_type}"

        context = {"field_name": field_name}

        if expected_type:
            context["expected_type"] = expected_type

        if actual_value is not None:
            context["actual_value"] = repr(actual_value)

        # Add any additional context parameters
        context.update(additional_context)

        return cls(message=message, context=context)


class DataTransformationError(DataError):
    """Raised when data transformation fails."""

    DEFAULT_ERROR_CODE = 4002
    DEFAULT_RECOVERY_HINT = "Check transformation parameters and input data format"


class DataLoadError(DataError):
    """Raised when loading data fails."""

    DEFAULT_ERROR_CODE = 4003
    DEFAULT_RECOVERY_HINT = "Check data source access and format"


class DatasetNotFoundError(DataError):
    """Raised when a requested dataset is not found."""

    DEFAULT_ERROR_CODE = 4004
    DEFAULT_RECOVERY_HINT = "Verify dataset name and ensure it's available"

    @classmethod
    def for_dataset(cls, dataset_name: str) -> "DatasetNotFoundError":
        """Create an exception for a specific dataset not found.

        Args:
            dataset_name: Name of the dataset that wasn't found

        Returns:
            A new DatasetNotFoundError
        """
        message = f"Dataset '{dataset_name}' not found"
        return cls(message=message, context={"dataset_name": dataset_name})


class GatedDatasetAuthenticationError(DataError):
    """Raised when authentication is required for a gated dataset."""

    DEFAULT_ERROR_CODE = 4005
    DEFAULT_RECOVERY_HINT = "Authenticate with the dataset provider"

    @classmethod
    def for_huggingface_dataset(
        cls, dataset_name: str
    ) -> "GatedDatasetAuthenticationError":
        """Create an exception for a gated HuggingFace dataset requiring authentication.

        Args:
            dataset_name: Name of the gated dataset

        Returns:
            A new GatedDatasetAuthenticationError with detailed guidance
        """
        message = (
            f"Dataset '{dataset_name}' is a gated dataset requiring authentication. "
            "You must authenticate with HuggingFace before accessing this dataset."
        )

        recovery_steps = (
            "Run `huggingface-cli login` to authenticate with your HuggingFace account. "
            "If you don't have access to this dataset, you may need to request access "
            "from the dataset owner through the HuggingFace Hub web interface."
        )

        return cls(
            message=message,
            context={
                "dataset_name": dataset_name,
                "provider": "huggingface",
                "auth_command": "huggingface-cli login",
                "recovery_steps": recovery_steps,
            },
            recovery_hint=recovery_steps,
        )


# =========================================================
# XCS Framework Exceptions (5000-5999)
# =========================================================


class XCSContext(ExceptionContext):
    """Type for XCS exception context."""

    node_id: Optional[str]
    graph_id: Optional[str]
    transform_name: Optional[str]


class XCSError(EmberError):
    """Base class for XCS-related errors."""

    DEFAULT_ERROR_CODE = 5000
    DEFAULT_RECOVERY_HINT = "Check XCS configuration and execution environment"


class TraceError(XCSError):
    """Raised when an error occurs during tracing operations."""

    DEFAULT_ERROR_CODE = 5001
    DEFAULT_RECOVERY_HINT = "Check function tracing and graph construction"

    @classmethod
    def during_trace(
        cls,
        operation_id: Optional[str] = None,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> "TraceError":
        """Create an exception for a specific tracing error.

        Args:
            operation_id: Optional ID of the tracing operation
            message: Optional custom message
            cause: Optional exception that caused this error

        Returns:
            A new TraceError
        """
        if message is None:
            message = "Error during execution tracing"
            if operation_id:
                message += f" of operation '{operation_id}'"

        context = {}
        if operation_id:
            context["operation_id"] = operation_id

        return cls(message=message, context=context, cause=cause)


class CompilationError(XCSError):
    """Raised when an error occurs during graph compilation."""

    DEFAULT_ERROR_CODE = 5002
    DEFAULT_RECOVERY_HINT = "Check graph structure and node definitions"

    @classmethod
    def for_graph(
        cls,
        graph_id: Optional[str] = None,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> "CompilationError":
        """Create an exception for a specific graph compilation error.

        Args:
            graph_id: Optional ID of the graph
            message: Optional custom message
            cause: Optional exception that caused this error

        Returns:
            A new CompilationError
        """
        if message is None:
            message = "Error during graph compilation"
            if graph_id:
                message += f" for graph '{graph_id}'"

        context = {}
        if graph_id:
            context["graph_id"] = graph_id

        return cls(message=message, context=context, cause=cause)


class ExecutionError(XCSError):
    """Raised when an error occurs during graph execution."""

    DEFAULT_ERROR_CODE = 5003
    DEFAULT_RECOVERY_HINT = "Check node inputs and execution environment"

    @classmethod
    def for_node(
        cls,
        node_id: Optional[str] = None,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context: Any,
    ) -> "ExecutionError":
        """Create an exception for a specific node execution error.

        Args:
            node_id: Optional ID of the node
            message: Optional custom message
            cause: Optional exception that caused this error
            **context: Additional context fields

        Returns:
            A new ExecutionError
        """
        if message is None:
            message = "Error during graph execution"
            if node_id:
                message += f" in node '{node_id}'"
            if cause:
                message += f": {str(cause)}"

        context_dict = dict(context)
        if node_id:
            context_dict["node_id"] = node_id

        return cls(message=message, context=context_dict, cause=cause)


class TransformError(XCSError):
    """Raised when an error occurs with XCS transforms."""

    DEFAULT_ERROR_CODE = 5010
    DEFAULT_RECOVERY_HINT = "Check transform parameters and input data"

    @classmethod
    def for_transform(
        cls,
        transform_name: Optional[str] = None,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
        **context: Any,
    ) -> "TransformError":
        """Create an exception for a specific transform error.

        Args:
            transform_name: Optional name of the transform
            message: Optional custom message
            cause: Optional exception that caused this error
            details: Optional detailed information about the error
            **context: Additional context fields

        Returns:
            A new TransformError
        """
        if message is None:
            message = "Error in XCS transform"
            if transform_name:
                message += f" '{transform_name}'"
            if cause:
                message += f": {str(cause)}"

        context_dict = dict(context)
        if transform_name:
            context_dict["transform_name"] = transform_name

        if details:
            context_dict.update(details)

        return cls(message=message, context=context_dict, cause=cause)


class ParallelExecutionError(ExecutionError):
    """Raised when parallel execution fails."""

    DEFAULT_ERROR_CODE = 5011
    DEFAULT_RECOVERY_HINT = "Check concurrency settings and worker availability"

    @classmethod
    def for_worker(
        cls,
        worker_id: str,
        node_id: Optional[str] = None,
        message: Optional[str] = None,
        cause: Optional[Exception] = None,
        **context: Any,
    ) -> "ParallelExecutionError":
        """Create an exception for a specific worker execution error.

        Args:
            worker_id: ID of the worker
            node_id: Optional ID of the node
            message: Optional custom message
            cause: Optional exception that caused this error
            **context: Additional context fields

        Returns:
            A new ParallelExecutionError
        """
        if message is None:
            message = f"Error during parallel execution in worker '{worker_id}'"
            if node_id:
                message += f" for node '{node_id}'"
            if cause:
                message += f": {str(cause)}"

        context_dict = dict(context)
        context_dict["worker_id"] = worker_id
        if node_id:
            context_dict["node_id"] = node_id

        return cls(message=message, context=context_dict, cause=cause)


class DataFlowError(XCSError):
    """Raised when there is an error in data flow analysis or processing."""

    DEFAULT_ERROR_CODE = 5020
    DEFAULT_RECOVERY_HINT = "Check node connections and data flow paths"

    @classmethod
    def for_connection(
        cls,
        source_node: str,
        target_node: str,
        graph_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> "DataFlowError":
        """Create an exception for a specific data flow error.

        Args:
            source_node: ID of the source node
            target_node: ID of the target node
            graph_id: Optional ID of the graph
            message: Optional custom message

        Returns:
            A new DataFlowError
        """
        if message is None:
            message = (
                f"Data flow error between nodes '{source_node}' and '{target_node}'"
            )
            if graph_id:
                message += f" in graph '{graph_id}'"

        context = {"source_node": source_node, "target_node": target_node}

        if graph_id:
            context["graph_id"] = graph_id

        return cls(message=message, context=context)


class SchedulerError(XCSError):
    """Raised when there is an error in the XCS execution scheduler."""

    DEFAULT_ERROR_CODE = 5030
    DEFAULT_RECOVERY_HINT = "Check scheduler configuration and node dependencies"

    @classmethod
    def for_scheduler(
        cls,
        scheduler_type: str,
        graph_id: Optional[str] = None,
        message: Optional[str] = None,
    ) -> "SchedulerError":
        """Create an exception for a specific scheduler error.

        Args:
            scheduler_type: Type of the scheduler
            graph_id: Optional ID of the graph
            message: Optional custom message

        Returns:
            A new SchedulerError
        """
        if message is None:
            message = f"Error in {scheduler_type} scheduler"
            if graph_id:
                message += f" for graph '{graph_id}'"

        context = {"scheduler_type": scheduler_type}

        if graph_id:
            context["graph_id"] = graph_id

        return cls(message=message, context=context)


# =========================================================
# API Exceptions (7000-7999)
# =========================================================


class APIError(EmberError):
    """Base class for API-related errors."""

    DEFAULT_ERROR_CODE = 7000
    DEFAULT_RECOVERY_HINT = "Check API usage and parameters"


# =========================================================
# CLI Exceptions (8000-8999)
# =========================================================


class CLIError(EmberError):
    """Base class for CLI-related errors."""

    DEFAULT_ERROR_CODE = 8000
    DEFAULT_RECOVERY_HINT = "Check command syntax and arguments"


# =========================================================
# Plugin System Exceptions (9000-9999)
# =========================================================


class PluginError(EmberError):
    """Base class for plugin-related errors."""

    DEFAULT_ERROR_CODE = 9000
    DEFAULT_RECOVERY_HINT = "Check plugin configuration and compatibility"


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not found."""

    DEFAULT_ERROR_CODE = 9001
    DEFAULT_RECOVERY_HINT = "Verify plugin name and ensure it's installed"

    @classmethod
    def for_plugin(cls, plugin_name: str) -> "PluginNotFoundError":
        """Create an exception for a specific plugin not found.

        Args:
            plugin_name: Name of the plugin that wasn't found

        Returns:
            A new PluginNotFoundError
        """
        message = f"Plugin '{plugin_name}' not found"
        return cls(message=message, context={"plugin_name": plugin_name})


class PluginConfigError(PluginError):
    """Raised when there's an issue with plugin configuration."""

    DEFAULT_ERROR_CODE = 9002
    DEFAULT_RECOVERY_HINT = "Check plugin configuration settings"


class PluginLoadError(PluginError):
    """Raised when loading a plugin fails."""

    DEFAULT_ERROR_CODE = 9003
    DEFAULT_RECOVERY_HINT = "Check plugin dependencies and environment"

    @classmethod
    def for_plugin(
        cls, plugin_name: str, reason: str, cause: Optional[Exception] = None
    ) -> "PluginLoadError":
        """Create an exception for a specific plugin load error.

        Args:
            plugin_name: Name of the plugin
            reason: Reason for the load failure
            cause: Optional exception that caused this error

        Returns:
            A new PluginLoadError
        """
        message = f"Failed to load plugin '{plugin_name}': {reason}"
        return cls(
            message=message,
            context={"plugin_name": plugin_name, "reason": reason},
            cause=cause,
        )


# =========================================================
# Special Purpose Exceptions
# =========================================================


class EmberWarning(Warning):
    """Base class for all Ember warnings."""

    pass


class DeprecationWarning(EmberWarning):
    """Warning about deprecated features."""

    pass


# =========================================================
# Legacy/Compatibility Layer
# =========================================================

# Aliases for backward compatibility with existing code
PromptSpecificationError = InvalidPromptError
OperatorSpecificationNotDefinedError = OperatorSpecificationError
BoundMethodNotInitializedError = InitializationError
EmberException = EmberError  # From operator exceptions.py
ConfigurationError = ConfigError  # For backward compatibility


# Export all exception classes for easy importing
__all__ = [
    # Base classes
    "EmberError",
    "EmberWarning",
    "DeprecationWarning",
    "ErrorGroup",
    # Type definitions
    "ExceptionContext",
    "ModelContext",
    "OperatorContext",
    "DataContext",
    "XCSContext",
    # Core exceptions
    "InvalidArgumentError",
    "ValidationError",
    "NotImplementedFeatureError",
    "DeprecationError",
    "IncompatibleTypeError",
    "InitializationError",
    # Registry exceptions
    "RegistryError",
    "ItemNotFoundError",
    "DuplicateItemError",
    "RegistrationError",
    # Configuration exceptions
    "ConfigError",
    "ConfigValidationError",
    "ConfigFileError",
    "ConfigValueError",
    "MissingConfigError",
    "ConfigurationError",  # Legacy alias
    # Model exceptions
    "ModelError",
    "ModelProviderError",
    "ModelNotFoundError",
    "ProviderAPIError",
    "ProviderConfigError",
    "ModelDiscoveryError",
    "ModelRegistrationError",
    "MissingLMModuleError",
    "InvalidPromptError",
    # Operator exceptions
    "OperatorError",
    "OperatorSpecificationError",
    "OperatorExecutionError",
    "SpecificationValidationError",
    "TreeTransformationError",
    "FlattenError",
    # Data exceptions
    "DataError",
    "DataValidationError",
    "DataTransformationError",
    "DataLoadError",
    "DatasetNotFoundError",
    # XCS exceptions
    "XCSError",
    "TraceError",
    "CompilationError",
    "ExecutionError",
    "TransformError",
    "ParallelExecutionError",
    "DataFlowError",
    "SchedulerError",
    # API exceptions
    "APIError",
    # CLI exceptions
    "CLIError",
    # Plugin exceptions
    "PluginError",
    "PluginNotFoundError",
    "PluginConfigError",
    "PluginLoadError",
    # Legacy compatibility
    "PromptSpecificationError",
    "OperatorSpecificationNotDefinedError",
    "BoundMethodNotInitializedError",
    "EmberException",
]
