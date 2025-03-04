/**
 * Error Handler Utilities
 * 
 * This module provides utilities for handling, formatting, and displaying errors
 * with a consistent, user-friendly appearance.
 */

import chalk from 'chalk';
import figures from 'figures';
import { getOutputOptions } from '../ui/output-manager';
import { getFallbackChar, supportsUnicode } from '../utils/terminal';
import { EmberCliError } from './ember-errors';
import { formatBox, formatUrl } from '../ui/components';

/**
 * Format options for error display
 */
export interface ErrorFormatOptions {
  /** Whether to include the error code */
  includeCode?: boolean;
  
  /** Whether to include suggestions */
  includeSuggestions?: boolean;
  
  /** Whether to include documentation links */
  includeDocsLinks?: boolean;
  
  /** Whether to include the stack trace (if available) */
  includeStack?: boolean;
  
  /** Whether to use color in the output */
  useColor?: boolean;
  
  /** Whether to format as JSON */
  asJson?: boolean;
  
  /** Whether to use detailed formatting with boxes */
  detailed?: boolean;
}

/**
 * Format an error into a string or JSON for display
 * 
 * @param error Error to format
 * @param options Formatting options
 * @returns Formatted error as string or JSON object
 */
export function formatError(
  error: unknown, 
  options: ErrorFormatOptions = {}
): string | object {
  const outputOptions = getOutputOptions();
  
  // Set default options
  const opts = {
    includeCode: true,
    includeSuggestions: true,
    includeDocsLinks: true,
    includeStack: process.env.DEBUG === 'true',
    useColor: !outputOptions.noColor,
    asJson: outputOptions.format === 'json',
    detailed: true,
    ...options
  };
  
  // Convert to EmberCliError if not already
  let emberError: EmberCliError;
  
  if (error instanceof EmberCliError) {
    emberError = error;
  } else if (error instanceof Error) {
    emberError = new EmberCliError(error.message, undefined, { cause: error });
  } else if (typeof error === 'string') {
    emberError = new EmberCliError(error);
  } else {
    const errorString = String(error);
    emberError = new EmberCliError(errorString || 'An unknown error occurred');
  }
  
  // Format as JSON if requested
  if (opts.asJson) {
    const jsonError: Record<string, any> = {
      error: {
        code: emberError.code,
        name: emberError.name,
        message: emberError.message,
      }
    };
    
    if (opts.includeSuggestions && emberError.suggestions?.length) {
      jsonError.error.suggestions = emberError.suggestions;
    }
    
    if (opts.includeDocsLinks && emberError.docsLinks?.length) {
      jsonError.error.documentation = emberError.docsLinks;
    }
    
    if (opts.includeStack && emberError.stack) {
      jsonError.error.stack = emberError.stack.split('\n');
    }
    
    if (emberError.context) {
      jsonError.error.context = emberError.context;
    }
    
    return jsonError;
  }
  
  // Format as string with optional color
  const c = opts.useColor ? chalk : { 
    red: (s: string) => s, 
    yellow: (s: string) => s, 
    cyan: (s: string) => s,
    dim: (s: string) => s,
    bold: { red: (s: string) => s }
  };
  
  const errorSymbol = supportsUnicode() ? figures.cross : getFallbackChar('✖');
  let output = '';
  
  // Detailed formatting with boxes and better visual structure
  if (opts.detailed) {
    // Error header with code
    output += c.bold.red(`${errorSymbol} Error`);
    if (opts.includeCode) {
      output += c.red(` [${emberError.code}]`);
    }
    output += '\n';
    
    // Error message
    output += c.red(emberError.message) + '\n';
    
    // Add suggestions in a formatted box
    if (opts.includeSuggestions && emberError.suggestions?.length) {
      const suggestionsText = emberError.suggestions.map(s => `• ${s}`).join('\n');
      output += '\n' + formatBox(suggestionsText, {
        title: 'Suggestions',
        titleColor: c.yellow,
        borderColor: c.yellow.dim,
        padding: 1,
      }) + '\n';
    }
    
    // Add documentation links
    if (opts.includeDocsLinks && emberError.docsLinks?.length) {
      output += '\n' + c.cyan('Documentation:');
      emberError.docsLinks.forEach(link => {
        output += `\n• ${formatUrl(link)}`;
      });
      output += '\n';
    }
    
    // Add stack trace
    if (opts.includeStack && emberError.stack) {
      output += '\n' + c.dim('Stack trace:') + '\n';
      output += c.dim(emberError.stack.split('\n').slice(1).join('\n')) + '\n';
    }
  } 
  // Simple formatting for compact display
  else {
    // Add error code and message
    if (opts.includeCode) {
      output += c.red(`[Error ${emberError.code}] `);
    } else {
      output += c.red(`${errorSymbol} Error: `);
    }
    
    output += c.red(emberError.message);
    
    // Add suggestions
    if (opts.includeSuggestions && emberError.suggestions?.length) {
      output += '\n\n';
      output += 'Suggestions:';
      emberError.suggestions.forEach(suggestion => {
        output += `\n${c.yellow('• ')}${suggestion}`;
      });
    }
    
    // Add documentation links
    if (opts.includeDocsLinks && emberError.docsLinks?.length) {
      output += '\n\n';
      output += 'Documentation:';
      emberError.docsLinks.forEach(link => {
        output += `\n• ${formatUrl(link)}`;
      });
    }
    
    // Add stack trace
    if (opts.includeStack && emberError.stack) {
      output += '\n\n';
      output += c.dim(emberError.stack.split('\n').slice(1).join('\n'));
    }
  }
  
  return output;
}

/**
 * Handle an error by formatting and printing it
 * 
 * @param error Error to handle
 * @param options Formatting options
 * @param exitProcess Whether to exit the process
 * @param exitCode Custom exit code to use
 */
export function handleError(
  error: unknown, 
  options: ErrorFormatOptions = {},
  exitProcess = false,
  exitCode?: number
): void {
  const outputOptions = getOutputOptions();
  
  // Skip in quiet mode unless it's a critical error
  if (outputOptions.quiet && !exitProcess) {
    return;
  }
  
  // Format the error
  const formattedError = formatError(error, options);
  
  // Print the error
  if (typeof formattedError === 'string') {
    console.error(formattedError);
  } else {
    console.error(JSON.stringify(formattedError, null, 2));
  }
  
  // Exit if requested
  if (exitProcess) {
    // Use provided exit code or determine one based on the error
    if (exitCode !== undefined) {
      process.exit(exitCode);
    }
    
    let code = 1;
    
    if (error instanceof EmberCliError) {
      // Use the error code's last two digits as exit code (max 125 to avoid conflict with signal codes)
      code = Math.min(error.code % 100, 125);
      if (code === 0) code = 1; // Ensure non-zero exit code
    }
    
    process.exit(code);
  }
}

/**
 * Try to execute a function and handle any errors
 * 
 * @param fn Function to execute
 * @param errorOptions Formatting options for any errors
 * @param exitOnError Whether to exit the process on error
 * @returns Result of the function or null if an error occurred
 */
export async function tryCatch<T>(
  fn: () => Promise<T> | T,
  errorOptions: ErrorFormatOptions = {},
  exitOnError = false
): Promise<T | null> {
  try {
    return await fn();
  } catch (error) {
    handleError(error, errorOptions, exitOnError);
    return null;
  }
}

/**
 * Assert that a condition is true, or throw an error
 * 
 * @param condition Condition to check
 * @param message Error message if condition is false
 * @param ErrorClass Error class to use
 * @param errorCode Error code to use
 */
export function assert(
  condition: any, 
  message: string, 
  ErrorClass: typeof EmberCliError = EmberCliError,
  errorCode?: number
): asserts condition {
  if (!condition) {
    throw new ErrorClass(message, errorCode);
  }
}