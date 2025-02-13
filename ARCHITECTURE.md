# Ember Core Architecture

This document provides a high-level overview of the Ember core architecture, its main components, extension points, and how dependency injection has been employed to improve testability and flexibility.

## High-Level Components

- **EmberAppContext:**  
  The central application context responsible for aggregating and providing:
  - `ConfigManager` (for config, environment variable resolution)
  - `ModelRegistry` (handles discovery and registration of models)
  - `UsageService` (tracks API usage across calls)
  - A common `logger` instance

- **ConfigCore:**  
  Centralized logic for merging configuration files, reading from environment variables, and validating settings. All configuration-related modules should delegate to this core.

- **ModelService & Registry:**  
  `ModelService` acts as a high-level facade that pulls a model from the `ModelRegistry`, delegates API calls, and logs usage via `UsageService`. The registry itself is populated during application initialization using auto-discovery routines.

- **Provider Extensions & Discovery:**  
  New providers should subclass `BaseProviderModel` and be registered via a unified ModelFactory. See the EXTENDING.md guide for step-by-step instructions.

## Data Flow Diagram

## Core Components

*   **`ember.core.app_context`**: Contains the `EmberAppContext` class, which holds references to core services like `ConfigManager`, `ModelRegistry`, and `UsageService`.  This is the central point of access for the application.
*   **`ember.core.configs`**: Manages configuration loading and access. The `ConfigManager` class handles reading configuration files, merging settings, and providing access to configuration values.
*   **`ember.core.registry`**:  Handles the registration and instantiation of models.
    *   `model`: Contains the core logic for model management, including the `ModelRegistry`, `ModelService`, `ModelFactory`, and provider-specific implementations.
    *   `provider_registry`:  Contains base classes and specific implementations for different model providers (e.g., OpenAI, Anthropic, Deepmind).
*   **`ember.core.exceptions`**: Defines custom exception classes used throughout the library for consistent error handling.
*   **`ember.core.non`**: 

## Model Invocation Flow

When a language model call is made, the following sequence of events occurs:

1.  The user interacts with the `ModelService`, either directly or through a higher-level operator.
2.  `ModelService` uses the provided model ID to retrieve the corresponding `BaseProviderModel` instance from the `ModelRegistry`.
3.  `ModelService` calls the `__call__` method of the `BaseProviderModel` instance, passing the prompt and any additional parameters.
4.  The `BaseProviderModel` implementation handles the provider-specific API call and returns a `ChatResponse`.
5.  If a `UsageService` is configured, `ModelService` logs usage statistics from the `ChatResponse`.

## Package Structure

| Directory                     | Description                                                                                                                                                                                                                                                           |
| :---------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ember.core.configs`          | Handles configuration loading, merging, and environment variable resolution.                                                                                                                                                                                        |
| `ember.core.registry.model`   | Manages the registration, instantiation, and invocation of models. Includes core schemas, services (ModelService, UsageService), the ModelRegistry, ModelFactory, and provider-specific implementations.                                                       |
| `ember.core.registry.operator`| (Further description needed)                                                                                                                                                                                                                                        |
| `ember.core.exceptions`       | Defines custom exception classes for consistent error handling.                                                                                                                                                                                                     |
| `ember.core.utils`            | (Further description needed - contains data and eval base classes)                                                                                                                                                                                |
| `ember.core.non`              | (Further description needed - this seems to be a set of higher-level operators, potentially a "non-differentiable" module, but clarification is needed on the name and purpose)                                                                                     |
| `ember.examples`              | Contains example scripts demonstrating various usage patterns of the library.                                                                                                                                                                                        |

For a deeper dive into the model registry and the "network-of-networks" approach, see [model_readme.md](model_readme.md). 