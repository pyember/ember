import ora, { Ora } from 'ora';
import chalk from 'chalk';
import logSymbols from 'log-symbols';
import cliProgress from 'cli-progress';
import figures from 'figures';
import cliTruncate from 'cli-truncate';
import wrapAnsi from 'wrap-ansi';
import { getTerminalColumns } from '../utils/terminal';

/**
 * UI Component Library for Ember CLI
 * 
 * Provides standardized UI components with consistent styling
 * and behavior for use across the CLI application.
 */

/**
 * Configuration options for UI components
 */
export interface UIConfig {
  /** Whether to disable animations and interactivity */
  quiet: boolean;
  /** Whether to disable colors */
  noColor: boolean;
  /** Whether to enable verbose output */
  verbose: boolean;
  /** Whether to enable JSON output */
  json: boolean;
}

/**
 * Creates a default UI configuration
 */
export function createDefaultUIConfig(): UIConfig {
  return {
    quiet: false,
    noColor: false,
    verbose: false,
    json: false
  };
}

/**
 * Global UI configuration
 */
let globalConfig: UIConfig = createDefaultUIConfig();

/**
 * Sets the global UI configuration
 */
export function setUIConfig(config: Partial<UIConfig>): void {
  globalConfig = { ...globalConfig, ...config };
}

/**
 * Gets the current UI configuration
 */
export function getUIConfig(): UIConfig {
  return { ...globalConfig };
}

/**
 * Standard color palette
 */
export const colors = {
  primary: (text: string) => globalConfig.noColor ? text : chalk.cyan(text),
  secondary: (text: string) => globalConfig.noColor ? text : chalk.blue(text),
  success: (text: string) => globalConfig.noColor ? text : chalk.green(text),
  warning: (text: string) => globalConfig.noColor ? text : chalk.yellow(text),
  error: (text: string) => globalConfig.noColor ? text : chalk.red(text),
  info: (text: string) => globalConfig.noColor ? text : chalk.white(text),
  dim: (text: string) => globalConfig.noColor ? text : chalk.dim(text),
  highlight: (text: string) => globalConfig.noColor ? text : chalk.bold.cyan(text),
  header: (text: string) => globalConfig.noColor ? text : chalk.bold.yellow(text),
  command: (text: string) => globalConfig.noColor ? text : chalk.bold.magenta(text),
  param: (text: string) => globalConfig.noColor ? text : chalk.italic.cyan(text),
  link: (text: string) => globalConfig.noColor ? text : chalk.underline.blue(text),
  code: (text: string) => globalConfig.noColor ? text : chalk.gray.italic(text)
};

/**
 * Standard symbols
 */
export const symbols = {
  success: globalConfig.noColor ? '✓' : logSymbols.success,
  error: globalConfig.noColor ? '✖' : logSymbols.error,
  warning: globalConfig.noColor ? '⚠' : logSymbols.warning,
  info: globalConfig.noColor ? 'ℹ' : logSymbols.info,
  pending: globalConfig.noColor ? '○' : figures.circle,
  bullet: globalConfig.noColor ? '•' : figures.bullet,
  pointer: globalConfig.noColor ? '>' : figures.pointer,
  ellipsis: globalConfig.noColor ? '...' : figures.ellipsis,
  arrowRight: globalConfig.noColor ? '->' : figures.arrowRight,
  questionMark: globalConfig.noColor ? '?' : figures.questionMarkPrefix
};

/**
 * Spinner wrapper class with enhanced features
 */
export class Spinner {
  private spinner: Ora | null = null;
  private isActive = false;
  private startTime = 0;
  
  /**
   * Creates a new spinner
   * @param text Initial spinner text
   */
  constructor(private text: string = '') {}
  
  /**
   * Starts the spinner
   * @param text Optional text to display
   * @returns this for chaining
   */
  start(text?: string): Spinner {
    if (globalConfig.quiet || globalConfig.json) {
      return this;
    }
    
    if (text) {
      this.text = text;
    }
    
    this.spinner = ora({
      text: this.text,
      color: 'cyan',
      spinner: 'dots'
    }).start();
    
    this.isActive = true;
    this.startTime = Date.now();
    return this;
  }
  
  /**
   * Updates the spinner text
   * @param text New text to display
   * @returns this for chaining
   */
  update(text: string): Spinner {
    if (this.isActive && this.spinner) {
      this.text = text;
      this.spinner.text = text;
    }
    return this;
  }
  
  /**
   * Stops the spinner with a success message
   * @param text Success message
   * @returns this for chaining
   */
  succeed(text?: string): Spinner {
    if (this.isActive && this.spinner) {
      const duration = this.getDurationText();
      this.spinner.succeed(text ? `${text} ${duration}` : `${this.text} ${duration}`);
      this.isActive = false;
    } else if (!globalConfig.quiet && !globalConfig.json) {
      console.log(`${symbols.success} ${text || this.text}`);
    }
    return this;
  }
  
  /**
   * Stops the spinner with an error message
   * @param text Error message
   * @returns this for chaining
   */
  fail(text?: string): Spinner {
    if (this.isActive && this.spinner) {
      this.spinner.fail(text || this.text);
      this.isActive = false;
    } else if (!globalConfig.quiet && !globalConfig.json) {
      console.log(`${symbols.error} ${colors.error(text || this.text)}`);
    }
    return this;
  }
  
  /**
   * Stops the spinner with a warning message
   * @param text Warning message
   * @returns this for chaining
   */
  warn(text?: string): Spinner {
    if (this.isActive && this.spinner) {
      this.spinner.warn(text || this.text);
      this.isActive = false;
    } else if (!globalConfig.quiet && !globalConfig.json) {
      console.log(`${symbols.warning} ${colors.warning(text || this.text)}`);
    }
    return this;
  }
  
  /**
   * Stops the spinner with an info message
   * @param text Info message
   * @returns this for chaining
   */
  info(text?: string): Spinner {
    if (this.isActive && this.spinner) {
      this.spinner.info(text || this.text);
      this.isActive = false;
    } else if (!globalConfig.quiet && !globalConfig.json) {
      console.log(`${symbols.info} ${colors.info(text || this.text)}`);
    }
    return this;
  }
  
  /**
   * Stops the spinner
   * @returns this for chaining
   */
  stop(): Spinner {
    if (this.isActive && this.spinner) {
      this.spinner.stop();
      this.isActive = false;
    }
    return this;
  }
  
  /**
   * Whether the spinner is currently active
   */
  get active(): boolean {
    return this.isActive;
  }
  
  private getDurationText(): string {
    if (!this.isActive) return '';
    const duration = Date.now() - this.startTime;
    return colors.dim(`(${formatDuration(duration)})`);
  }
}

/**
 * Creates a new progress bar
 * @param options Progress bar options
 * @returns Progress bar instance
 */
export function createProgressBar(options?: {
  total?: number;
  width?: number;
  format?: string;
}): cliProgress.SingleBar {
  if (globalConfig.quiet || globalConfig.json) {
    // Return a no-op progress bar when in quiet or JSON mode
    const noop = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
    noop.start = () => noop;
    noop.update = () => noop;
    noop.increment = () => noop;
    noop.stop = () => {};
    return noop;
  }
  
  const format = options?.format || 
    `${colors.primary('Progress:')} [{bar}] ${colors.primary('{percentage}%')} | {value}/{total} | ${colors.dim('{duration_formatted}')}`;
  
  const bar = new cliProgress.SingleBar({
    format,
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: true,
    clearOnComplete: true,
    stopOnComplete: true,
    forceRedraw: true,
    barsize: options?.width || 40,
    etaBuffer: 10,
    noTTYOutput: globalConfig.quiet,
  });
  
  bar.start(options?.total || 100, 0);
  return bar;
}

/**
 * Message boxes for different types of notifications
 */
export const message = {
  /**
   * Displays a success message
   * @param text Message text
   */
  success(text: string): void {
    if (globalConfig.quiet || globalConfig.json) return;
    console.log(`${symbols.success} ${colors.success(text)}`);
  },
  
  /**
   * Displays an error message
   * @param text Message text
   */
  error(text: string): void {
    if (globalConfig.json) return;
    console.error(`${symbols.error} ${colors.error(text)}`);
  },
  
  /**
   * Displays a warning message
   * @param text Message text
   */
  warning(text: string): void {
    if (globalConfig.quiet || globalConfig.json) return;
    console.warn(`${symbols.warning} ${colors.warning(text)}`);
  },
  
  /**
   * Displays an info message
   * @param text Message text
   */
  info(text: string): void {
    if (globalConfig.quiet || globalConfig.json) return;
    console.log(`${symbols.info} ${colors.info(text)}`);
  },
  
  /**
   * Displays a tip message
   * @param text Message text
   */
  tip(text: string): void {
    if (globalConfig.quiet || globalConfig.json) return;
    console.log(`${symbols.info} ${colors.primary(`TIP: ${text}`)}`);
  },
  
  /**
   * Displays a verbose message (only in verbose mode)
   * @param text Message text
   */
  verbose(text: string): void {
    if (!globalConfig.verbose || globalConfig.json) return;
    console.log(`${colors.dim(text)}`);
  },
  
  /**
   * Displays a debug message (only in verbose mode)
   * @param text Message text
   * @param data Optional data to display
   */
  debug(text: string, data?: any): void {
    if (!globalConfig.verbose || globalConfig.json) return;
    console.log(`${colors.dim(`[DEBUG] ${text}`)}`);
    if (data) {
      console.log(colors.dim(JSON.stringify(data, null, 2)));
    }
  }
};

/**
 * Formats text into a header
 * @param text Header text
 * @returns Formatted header
 */
export function formatHeader(text: string): string {
  if (globalConfig.json) return text;
  return `\n${colors.header(text)}\n${colors.dim('━'.repeat(Math.min(text.length, getTerminalColumns() - 10)))}\n`;
}

/**
 * Formats text into a subheader
 * @param text Subheader text
 * @returns Formatted subheader
 */
export function formatSubHeader(text: string): string {
  if (globalConfig.json) return text;
  return `\n${colors.secondary(text)}\n${colors.dim('─'.repeat(Math.min(text.length, getTerminalColumns() - 10)))}\n`;
}

/**
 * Formats a URL for display
 * @param url URL to format
 * @param text Optional text to display instead of the URL
 * @returns Formatted URL
 */
export function formatUrl(url: string, text?: string): string {
  if (globalConfig.json) return url;
  return colors.link(text || url);
}

/**
 * Formats a command for display
 * @param command Command to format
 * @returns Formatted command
 */
export function formatCommand(command: string): string {
  if (globalConfig.json) return command;
  return colors.command(command);
}

/**
 * Formats a parameter for display
 * @param param Parameter to format
 * @returns Formatted parameter
 */
export function formatParam(param: string): string {
  if (globalConfig.json) return param;
  return colors.param(param);
}

/**
 * Formats code for display
 * @param code Code to format
 * @returns Formatted code
 */
export function formatCode(code: string): string {
  if (globalConfig.json) return code;
  return colors.code(`\`${code}\``);
}

/**
 * Formats text into a box
 * @param text Text to put in the box
 * @param options Box options
 * @returns Formatted box
 */
export function formatBox(text: string, options?: {
  padding?: number;
  title?: string;
  titleColor?: (text: string) => string;
  borderColor?: (text: string) => string;
}): string {
  if (globalConfig.json) return text;
  
  const padding = options?.padding || 1;
  const borderColor = options?.borderColor || colors.dim;
  const titleColor = options?.titleColor || colors.primary;
  
  const columns = getTerminalColumns();
  const wrappedText = wrapAnsi(text, columns - (padding * 2) - 4);
  const lines = wrappedText.split('\n');
  const width = Math.min(
    Math.max(...lines.map(line => line.length)) + (padding * 2),
    columns - 4
  );
  
  // Top border with optional title
  let result = borderColor('┌' + '─'.repeat(width + 2) + '┐') + '\n';
  
  // Title if provided
  if (options?.title) {
    const titleLine = ` ${options.title} `;
    const leftPadding = Math.floor((width + 2 - titleLine.length) / 2);
    const rightPadding = width + 2 - titleLine.length - leftPadding;
    
    result = borderColor('┌' + '─'.repeat(leftPadding)) + 
             titleColor(titleLine) + 
             borderColor('─'.repeat(rightPadding) + '┐') + '\n';
  }
  
  // Empty line for padding
  if (padding > 0) {
    result += borderColor('│') + ' '.repeat(width + 2) + borderColor('│') + '\n';
  }
  
  // Content
  lines.forEach(line => {
    const paddedLine = line.padEnd(width, ' ');
    result += borderColor('│') + ' ' + paddedLine + ' ' + borderColor('│') + '\n';
  });
  
  // Empty line for padding
  if (padding > 0) {
    result += borderColor('│') + ' '.repeat(width + 2) + borderColor('│') + '\n';
  }
  
  // Bottom border
  result += borderColor('└' + '─'.repeat(width + 2) + '┘');
  
  return result;
}

/**
 * Formats a duration in milliseconds to a human-readable string
 * @param ms Duration in milliseconds
 * @returns Formatted duration
 */
export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  } else if (ms < 60000) {
    return `${(ms / 1000).toFixed(1)}s`;
  } else {
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(1);
    return `${minutes}m ${seconds}s`;
  }
}

/**
 * Formats a data size in bytes to a human-readable string
 * @param bytes Size in bytes
 * @returns Formatted size
 */
export function formatSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  } else if (bytes < 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  } else {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }
}

/**
 * Truncates text to fit within a certain width
 * @param text Text to truncate
 * @param width Maximum width
 * @returns Truncated text
 */
export function truncate(text: string, width: number = getTerminalColumns() - 10): string {
  return cliTruncate(text, width);
}