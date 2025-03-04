/**
 * Tests for the error handling utilities
 */

import { 
  EmberCliError, 
  PythonBridgeError, 
  ModelError, 
  formatError, 
  GeneralErrorCodes, 
  ModelErrorCodes, 
  PythonBridgeErrorCodes 
} from '../../../src/cli/errors';

// Mock console.error
const originalConsoleError = console.error;
let consoleErrorMock: jest.Mock;

beforeEach(() => {
  consoleErrorMock = jest.fn();
  console.error = consoleErrorMock;
});

afterEach(() => {
  console.error = originalConsoleError;
});

describe('EmberCliError', () => {
  test('creates basic error with default code', () => {
    const error = new EmberCliError('Test error');
    expect(error.message).toBe('Test error');
    expect(error.code).toBe(GeneralErrorCodes.UNKNOWN_ERROR);
  });

  test('creates error with custom code and suggestions', () => {
    const error = new EmberCliError('Test error', 1234, {
      suggestions: ['Suggestion 1', 'Suggestion 2']
    });
    expect(error.message).toBe('Test error');
    expect(error.code).toBe(1234);
    expect(error.suggestions).toEqual(['Suggestion 1', 'Suggestion 2']);
  });

  test('getFullMessage formats error with suggestions', () => {
    const error = new EmberCliError('Test error', 1234, {
      suggestions: ['Suggestion 1', 'Suggestion 2']
    });
    
    const fullMessage = error.getFullMessage();
    expect(fullMessage).toContain('[Error 1234]');
    expect(fullMessage).toContain('Test error');
    expect(fullMessage).toContain('Suggestions:');
    expect(fullMessage).toContain('Suggestion 1');
    expect(fullMessage).toContain('Suggestion 2');
  });

  test('fromPythonError creates appropriate error type', () => {
    // ModelNotFoundError
    const error1 = EmberCliError.fromPythonError('ModelNotFoundError: Model gpt-4 not found');
    expect(error1).toBeInstanceOf(ModelError);
    expect(error1.code).toBe(ModelErrorCodes.MODEL_NOT_FOUND);
    expect(error1.message).toBe('Model gpt-4 not found');
    
    // Generic error
    const error2 = EmberCliError.fromPythonError('Something went wrong');
    expect(error2).toBeInstanceOf(EmberCliError);
    expect(error2.code).toBe(GeneralErrorCodes.UNKNOWN_ERROR);
    expect(error2.message).toBe('Something went wrong');
    
    // Python bridge error
    const error3 = EmberCliError.fromPythonError(
      'ModuleNotFoundError: No module named ember',
      'Default message',
      PythonBridgeErrorCodes.EMBER_IMPORT_ERROR
    );
    expect(error3).toBeInstanceOf(PythonBridgeError);
    expect(error3.message).toBe('No module named ember');
  });
});

describe('formatError', () => {
  test('formats EmberCliError with code and message', () => {
    const error = new EmberCliError('Test error', 1234);
    const formatted = formatError(error, { useColor: false });
    
    expect(formatted).toContain('[Error 1234]');
    expect(formatted).toContain('Test error');
  });
  
  test('formats standard Error', () => {
    const error = new Error('Standard error');
    const formatted = formatError(error, { useColor: false });
    
    expect(formatted).toContain('Error:');
    expect(formatted).toContain('Standard error');
  });
  
  test('formats error as JSON', () => {
    const error = new EmberCliError('Test error', 1234, {
      suggestions: ['Suggestion 1'],
      docsLinks: ['https://docs.example.com']
    });
    
    const formatted = formatError(error, { asJson: true });
    const parsed = JSON.parse(formatted);
    
    expect(parsed.code).toBe(1234);
    expect(parsed.message).toBe('Test error');
    expect(parsed.suggestions).toEqual(['Suggestion 1']);
    expect(parsed.documentation).toEqual(['https://docs.example.com']);
  });
  
  test('includes/excludes suggestions based on options', () => {
    const error = new EmberCliError('Test error', 1234, {
      suggestions: ['Suggestion 1', 'Suggestion 2']
    });
    
    const withSuggestions = formatError(error, { 
      useColor: false, 
      includeSuggestions: true 
    });
    expect(withSuggestions).toContain('Suggestions:');
    
    const withoutSuggestions = formatError(error, { 
      useColor: false, 
      includeSuggestions: false 
    });
    expect(withoutSuggestions).not.toContain('Suggestions:');
  });
});