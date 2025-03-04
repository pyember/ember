/**
 * Ember CLI Error Classes
 * 
 * This file defines all custom error classes used in the Ember CLI.
 * It provides a rich error hierarchy with detailed error information.
 */

import { 
  GeneralErrorCodes, 
  ModelErrorCodes, 
  ProjectErrorCodes, 
  PythonBridgeErrorCodes,
  PythonExceptionToErrorCode 
} from './error-codes';

/**
 * Base error class for all Ember CLI errors
 */
export class EmberCliError extends Error {
  /** Error code for identification */
  public code: number;
  
  /** Additional contextual information */
  public context?: Record<string, any>;
  
  /** Suggestions for fixing the error */
  public suggestions?: string[];
  
  /** Links to documentation */
  public docsLinks?: string[];
  
  constructor(message: string, code = GeneralErrorCodes.UNKNOWN_ERROR, options?: {
    cause?: Error,
    context?: Record<string, any>,
    suggestions?: string[],
    docsLinks?: string[]
  }) {
    super(message, { cause: options?.cause });
    this.name = 'EmberCliError';
    this.code = code;
    this.context = options?.context;
    this.suggestions = options?.suggestions;
    this.docsLinks = options?.docsLinks;
  }
  
  /**
   * Returns a full error message including code, message, suggestions and docs links
   */
  getFullMessage(): string {
    let fullMessage = `[Error ${this.code}] ${this.message}`;
    
    if (this.suggestions?.length) {
      fullMessage += '\n\nSuggestions:';
      this.suggestions.forEach(suggestion => {
        fullMessage += `\n- ${suggestion}`;
      });
    }
    
    if (this.docsLinks?.length) {
      fullMessage += '\n\nDocumentation:';
      this.docsLinks.forEach(link => {
        fullMessage += `\n- ${link}`;
      });
    }
    
    return fullMessage;
  }
  
  /**
   * Creates an error instance from a Python exception
   */
  static fromPythonError(
    error: any, 
    defaultMessage = 'An unknown error occurred',
    defaultCode = GeneralErrorCodes.UNKNOWN_ERROR
  ): EmberCliError {
    // Extract Python exception name and message
    const match = error?.match?.(/^([a-zA-Z0-9_]+)(?:\.[a-zA-Z0-9_.]+)*: (.+)$/);
    const exceptionType = match?.[1] || 'Exception';
    const message = match?.[2] || error?.toString() || defaultMessage;
    
    // Get the corresponding error code
    const code = PythonExceptionToErrorCode[exceptionType] || defaultCode;
    
    // Create the appropriate error instance based on code range
    if (code >= 3000 && code < 4000) {
      return new ModelError(message, code as ModelErrorCodes);
    } else if (code >= 4000 && code < 5000) {
      return new ProjectError(message, code as ProjectErrorCodes);
    } else if (code >= 2000 && code < 3000) {
      return new PythonBridgeError(message, code as PythonBridgeErrorCodes);
    } else {
      return new EmberCliError(message, code);
    }
  }
}

/**
 * Error class for Python bridge errors
 */
export class PythonBridgeError extends EmberCliError {
  constructor(
    message: string, 
    code = PythonBridgeErrorCodes.BRIDGE_INITIALIZATION_FAILED,
    options?: {
      cause?: Error,
      context?: Record<string, any>,
      suggestions?: string[],
      docsLinks?: string[]
    }
  ) {
    super(message, code, options);
    this.name = 'PythonBridgeError';
    
    // Add default suggestions if none provided
    if (!options?.suggestions) {
      this.suggestions = [
        'Ensure Python is installed and in your PATH',
        'Verify that the Ember package is installed (`pip install ember-ai`)',
        'Check for any Python error messages in the output'
      ];
    }
  }
}

/**
 * Error class for model and provider errors
 */
export class ModelError extends EmberCliError {
  constructor(
    message: string, 
    code = ModelErrorCodes.MODEL_NOT_FOUND,
    options?: {
      cause?: Error,
      context?: Record<string, any>,
      suggestions?: string[],
      docsLinks?: string[]
    }
  ) {
    super(message, code, options);
    this.name = 'ModelError';
    
    // Add default suggestions based on error code
    if (!options?.suggestions) {
      switch (code) {
        case ModelErrorCodes.MODEL_NOT_FOUND:
          this.suggestions = [
            'Run `ember models` to list available models',
            'Check that the provider is properly configured',
            'Verify that the model name is spelled correctly'
          ];
          break;
        case ModelErrorCodes.PROVIDER_NOT_FOUND:
          this.suggestions = [
            'Run `ember providers` to list available providers',
            'Ensure the provider is properly installed',
            'Check your spelling of the provider name'
          ];
          break;
        case ModelErrorCodes.PROVIDER_API_ERROR:
        case ModelErrorCodes.PROVIDER_CONFIG_ERROR:
          this.suggestions = [
            'Verify your API keys are correctly set',
            'Check your provider configuration',
            'Ensure your account has access to the requested model'
          ];
          break;
      }
    }
  }
}

/**
 * Error class for project-related errors
 */
export class ProjectError extends EmberCliError {
  constructor(
    message: string, 
    code = ProjectErrorCodes.PROJECT_CREATION_FAILED,
    options?: {
      cause?: Error,
      context?: Record<string, any>,
      suggestions?: string[],
      docsLinks?: string[]
    }
  ) {
    super(message, code, options);
    this.name = 'ProjectError';
    
    // Add default suggestions based on error code
    if (!options?.suggestions) {
      switch (code) {
        case ProjectErrorCodes.PROJECT_EXISTS:
          this.suggestions = [
            'Choose a different project name',
            'Delete the existing directory first if you want to overwrite it',
            'Use the --force option to overwrite the existing project'
          ];
          break;
        case ProjectErrorCodes.PROJECT_NOT_FOUND:
          this.suggestions = [
            'Check that the project directory exists',
            'Verify the path to the project'
          ];
          break;
      }
    }
  }
}

/**
 * Error class for user input validation errors
 */
export class ValidationError extends EmberCliError {
  constructor(
    message: string, 
    code = GeneralErrorCodes.INVALID_ARGUMENT,
    options?: {
      cause?: Error,
      context?: Record<string, any>,
      suggestions?: string[],
      docsLinks?: string[]
    }
  ) {
    super(message, code, options);
    this.name = 'ValidationError';
  }
}

/**
 * Error class for configuration errors
 */
export class ConfigurationError extends EmberCliError {
  constructor(
    message: string, 
    code = GeneralErrorCodes.INVALID_CONFIG,
    options?: {
      cause?: Error,
      context?: Record<string, any>,
      suggestions?: string[],
      docsLinks?: string[]
    }
  ) {
    super(message, code, options);
    this.name = 'ConfigurationError';
    
    // Add default suggestions
    if (!options?.suggestions) {
      this.suggestions = [
        'Use `ember config validate` to check your configuration',
        'Reset to default with `ember config reset`',
        'Check the configuration file for syntax errors'
      ];
    }
  }
}

/**
 * Error class for authorization errors
 */
export class AuthorizationError extends EmberCliError {
  constructor(
    message: string, 
    code = GeneralErrorCodes.AUTH_REQUIRED,
    options?: {
      cause?: Error,
      context?: Record<string, any>,
      suggestions?: string[],
      docsLinks?: string[]
    }
  ) {
    super(message, code, options);
    this.name = 'AuthorizationError';
    
    // Add default suggestions
    if (!options?.suggestions) {
      this.suggestions = [
        'Set your API keys with `ember config set provider.apiKey YOUR_KEY`',
        'Check your environment variables for API keys',
        'Verify that your API key is valid and not expired'
      ];
    }
  }
}

/**
 * Error class for network-related errors
 */
export class NetworkError extends EmberCliError {
  constructor(
    message: string, 
    code = GeneralErrorCodes.NETWORK_ERROR,
    options?: {
      cause?: Error,
      context?: Record<string, any>,
      suggestions?: string[],
      docsLinks?: string[]
    }
  ) {
    super(message, code, options);
    this.name = 'NetworkError';
    
    // Add default suggestions
    if (!options?.suggestions) {
      this.suggestions = [
        'Check your internet connection',
        'Verify that the API service is available',
        'If using a proxy, ensure it\'s properly configured'
      ];
    }
  }
}