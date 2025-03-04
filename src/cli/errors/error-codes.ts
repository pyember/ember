/**
 * Error Codes for Ember CLI
 * 
 * This file defines all error codes used in Ember CLI.
 * Error codes are categorized by component and numbered sequentially.
 */

// General CLI errors (1000-1999)
export enum GeneralErrorCodes {
  // General errors
  UNKNOWN_ERROR = 1000,
  INITIALIZATION_FAILED = 1001,
  INVALID_ARGUMENT = 1002,
  MISSING_REQUIRED_ARGUMENT = 1003,
  FILE_NOT_FOUND = 1004,
  PERMISSION_DENIED = 1005,
  COMMAND_FAILED = 1006,
  OPERATION_TIMEOUT = 1007,
  
  // Configuration errors
  CONFIG_NOT_FOUND = 1100,
  CONFIG_PARSE_ERROR = 1101,
  CONFIG_WRITE_ERROR = 1102,
  INVALID_CONFIG = 1103,
  
  // Authentication errors
  AUTH_REQUIRED = 1200,
  AUTH_FAILED = 1201,
  API_KEY_MISSING = 1202,
  API_KEY_INVALID = 1203,
  
  // Network errors
  NETWORK_ERROR = 1300,
  SERVER_ERROR = 1301,
  REQUEST_FAILED = 1302,
  API_RATE_LIMIT = 1303,
}

// Python bridge errors (2000-2999)
export enum PythonBridgeErrorCodes {
  BRIDGE_INITIALIZATION_FAILED = 2000,
  PYTHON_EXECUTION_ERROR = 2001,
  PYTHON_NOT_FOUND = 2002,
  EMBER_IMPORT_ERROR = 2003,
  SERIALIZATION_ERROR = 2004,
  DESERIALIZATION_ERROR = 2005,
  EMBER_NOT_INSTALLED = 2006,
}

// Model and provider errors (3000-3999)
export enum ModelErrorCodes {
  MODEL_NOT_FOUND = 3000,
  PROVIDER_NOT_FOUND = 3001,
  MODEL_INVOCATION_FAILED = 3002,
  PROVIDER_API_ERROR = 3003,
  PROVIDER_CONFIG_ERROR = 3004,
  MODEL_VALIDATION_ERROR = 3005,
  INVALID_PROMPT = 3006,
  MODEL_DISCOVERY_ERROR = 3007,
  MODEL_REGISTRATION_ERROR = 3008,
}

// Project errors (4000-4999)
export enum ProjectErrorCodes {
  PROJECT_CREATION_FAILED = 4000,
  PROJECT_EXISTS = 4001,
  PROJECT_NOT_FOUND = 4002,
  PROJECT_VALIDATION_FAILED = 4003,
  INVALID_PROJECT_STRUCTURE = 4004,
}

// Map Python exception types to error codes
export const PythonExceptionToErrorCode: Record<string, number> = {
  // Base error classes
  'EmberError': GeneralErrorCodes.UNKNOWN_ERROR,
  'EmberException': GeneralErrorCodes.UNKNOWN_ERROR,
  
  // Provider errors
  'ProviderAPIError': ModelErrorCodes.PROVIDER_API_ERROR,
  'ProviderConfigError': ModelErrorCodes.PROVIDER_CONFIG_ERROR,
  
  // Model registry errors
  'ModelNotFoundError': ModelErrorCodes.MODEL_NOT_FOUND,
  'ModelRegistrationError': ModelErrorCodes.MODEL_REGISTRATION_ERROR,
  'ModelDiscoveryError': ModelErrorCodes.MODEL_DISCOVERY_ERROR,
  'RegistryError': GeneralErrorCodes.UNKNOWN_ERROR,
  
  // Validation errors
  'ValidationError': GeneralErrorCodes.INVALID_ARGUMENT,
  'InvalidPromptError': ModelErrorCodes.INVALID_PROMPT,
  
  // Configuration errors
  'ConfigurationError': GeneralErrorCodes.INVALID_CONFIG,
  
  // Operator errors
  'OperatorError': GeneralErrorCodes.COMMAND_FAILED,
  'OperatorExecutionError': GeneralErrorCodes.COMMAND_FAILED,
  'FlattenError': GeneralErrorCodes.COMMAND_FAILED,
  'OperatorSpecificationNotDefinedError': ModelErrorCodes.MODEL_VALIDATION_ERROR,
  'SpecificationValidationError': ModelErrorCodes.MODEL_VALIDATION_ERROR,
  'TreeTransformationError': GeneralErrorCodes.COMMAND_FAILED,
  
  // Prompt specification errors
  'PromptSpecificationError': ModelErrorCodes.INVALID_PROMPT,
  'PlaceholderMissingError': ModelErrorCodes.INVALID_PROMPT,
  'MismatchedModelError': ModelErrorCodes.MODEL_VALIDATION_ERROR,
  'InvalidInputTypeError': GeneralErrorCodes.INVALID_ARGUMENT,
  
  // Python standard errors
  'ValueError': GeneralErrorCodes.INVALID_ARGUMENT,
  'TypeError': GeneralErrorCodes.INVALID_ARGUMENT,
  'FileNotFoundError': GeneralErrorCodes.FILE_NOT_FOUND,
  'PermissionError': GeneralErrorCodes.PERMISSION_DENIED,
  'TimeoutError': GeneralErrorCodes.OPERATION_TIMEOUT,
  'ModuleNotFoundError': PythonBridgeErrorCodes.EMBER_IMPORT_ERROR,
};