/**
 * CLI options utilities
 * 
 * These utilities help with managing global CLI options.
 */

interface CliOptions {
  quiet: boolean;
  debug: boolean;
  color: boolean;
  json: boolean;
  compact: boolean;
  verbose: boolean;
}

/**
 * Get CLI options by parsing process.argv
 * This is a simple implementation that looks for specific flags
 * 
 * @returns CliOptions object with all option values
 */
export function getCliOptions(): CliOptions {
  const args = process.argv;
  
  return {
    quiet: args.includes('--quiet'),
    debug: args.includes('--debug'),
    color: !args.includes('--no-color'),
    json: args.includes('--json'),
    compact: args.includes('--compact'),
    verbose: args.includes('--verbose')
  };
}

/**
 * Check if JSON output is requested
 * 
 * @returns true if JSON output is requested
 */
export function isJsonOutput(): boolean {
  return getCliOptions().json;
}

/**
 * Check if we're in debug mode
 * 
 * @returns true if in debug mode
 */
export function isDebugMode(): boolean {
  return getCliOptions().debug;
}

/**
 * Check if we're in quiet mode
 * 
 * @returns true if in quiet mode
 */
export function isQuietMode(): boolean {
  return getCliOptions().quiet;
}

/**
 * Check if color output is enabled
 * 
 * @returns true if color output is enabled
 */
export function isColorEnabled(): boolean {
  return getCliOptions().color;
}

/**
 * Check if verbose output is enabled
 * 
 * @returns true if verbose output is enabled
 */
export function isVerboseMode(): boolean {
  return getCliOptions().verbose;
}

/**
 * Check if compact output is enabled
 * 
 * @returns true if compact output is enabled
 */
export function isCompactMode(): boolean {
  return getCliOptions().compact;
}