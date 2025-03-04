/**
 * Errors Module Index
 * 
 * This file exports all error-related utilities and classes for the CLI.
 */

// Export error codes
export * from './error-codes';

// Export error classes
export * from './ember-errors';

// Export error handling utilities
export * from './error-handler';

/**
 * Helper functions to create errors with proper typing
 */

import { 
  EmberCliError, 
  PythonBridgeError, 
  ModelError, 
  ProjectError, 
  ValidationError, 
  ConfigurationError,
  AuthorizationError,
  NetworkError
} from './ember-errors';

import {
  GeneralErrorCodes,
  PythonBridgeErrorCodes,
  ModelErrorCodes,
  ProjectErrorCodes
} from './error-codes';

/**
 * Create a general CLI error
 */
export function createError(
  message: string, 
  code = GeneralErrorCodes.UNKNOWN_ERROR,
  options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }
): EmberCliError {
  return new EmberCliError(message, code, options);
}

/**
 * Create a Python bridge error
 */
export function createPythonBridgeError(
  message: string,
  code = PythonBridgeErrorCodes.BRIDGE_INITIALIZATION_FAILED,
  options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }
): PythonBridgeError {
  return new PythonBridgeError(message, code, options);
}

/**
 * Create a model or provider error
 */
export function createModelError(
  message: string,
  code = ModelErrorCodes.MODEL_NOT_FOUND,
  options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }
): ModelError {
  return new ModelError(message, code, options);
}

/**
 * Create a project error
 */
export function createProjectError(
  message: string,
  code = ProjectErrorCodes.PROJECT_CREATION_FAILED,
  options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }
): ProjectError {
  return new ProjectError(message, code, options);
}

/**
 * Create a validation error
 */
export function createValidationError(
  message: string,
  code = GeneralErrorCodes.INVALID_ARGUMENT,
  options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }
): ValidationError {
  return new ValidationError(message, code, options);
}

/**
 * Create a configuration error
 */
export function createConfigError(
  message: string,
  code = GeneralErrorCodes.INVALID_CONFIG,
  options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }
): ConfigurationError {
  return new ConfigurationError(message, code, options);
}

/**
 * Create an authorization error
 */
export function createAuthError(
  message: string,
  code = GeneralErrorCodes.AUTH_REQUIRED,
  options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }
): AuthorizationError {
  return new AuthorizationError(message, code, options);
}

/**
 * Create a network error
 */
export function createNetworkError(
  message: string,
  code = GeneralErrorCodes.NETWORK_ERROR,
  options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }
): NetworkError {
  return new NetworkError(message, code, options);
}