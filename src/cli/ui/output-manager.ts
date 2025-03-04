import chalk from 'chalk';
import { getTerminalColumns, measureTextWidth } from '../utils/terminal';
import { UIConfig, getUIConfig } from './components';

/**
 * Output format type
 */
export type OutputFormat = 'text' | 'json';

/**
 * Options for output formatting
 */
export interface OutputOptions {
  /** Output format */
  format: OutputFormat;
  /** Whether to include extra details */
  verbose: boolean;
  /** Whether to suppress all output except errors */
  quiet: boolean;
  /** Whether to disable colors */
  noColor: boolean;
}

/**
 * Default output options
 */
const DEFAULT_OUTPUT_OPTIONS: OutputOptions = {
  format: 'text',
  verbose: false,
  quiet: false,
  noColor: false,
};

/**
 * Global current output options
 */
let currentOptions: OutputOptions = { ...DEFAULT_OUTPUT_OPTIONS };

/**
 * Sets global output options
 * @param options New output options
 */
export function setOutputOptions(options: Partial<OutputOptions>): void {
  currentOptions = { ...currentOptions, ...options };
  
  // Update UI config to match
  const uiConfig: UIConfig = {
    quiet: currentOptions.quiet,
    noColor: currentOptions.noColor,
    verbose: currentOptions.verbose,
    json: currentOptions.format === 'json',
  };
  
  // Update global UI config
  require('./components').setUIConfig(uiConfig);
}

/**
 * Gets current output options
 * @returns Current output options
 */
export function getOutputOptions(): OutputOptions {
  return { ...currentOptions };
}

/**
 * Output manager class
 * Handles consistent output formatting across the CLI
 */
export class OutputManager {
  private outputBuffer: string[] = [];
  private jsonOutput: Record<string, any> = {};
  private indentLevel = 0;
  private indentSize = 2;
  
  /**
   * Creates a new output manager
   * @param options Output options
   */
  constructor(private options: OutputOptions = currentOptions) {}
  
  /**
   * Gets an indentation string based on current indent level
   * @returns Indentation string
   */
  private getIndent(): string {
    return ' '.repeat(this.indentLevel * this.indentSize);
  }
  
  /**
   * Prints a line of text
   * @param text Text to print
   * @param jsonPath Optional JSON path to add this text to
   */
  print(text: string, jsonPath?: string): void {
    if (this.options.quiet && this.options.format !== 'json') {
      return;
    }
    
    const indent = this.getIndent();
    
    if (this.options.format === 'text') {
      this.outputBuffer.push(`${indent}${text}`);
    } else if (jsonPath && this.options.format === 'json') {
      this.setJsonValue(jsonPath, text);
    }
  }
  
  /**
   * Prints a line with newline
   * @param text Text to print
   * @param jsonPath Optional JSON path to add this text to
   */
  println(text: string = '', jsonPath?: string): void {
    this.print(text, jsonPath);
    if (this.options.format === 'text') {
      this.outputBuffer.push('');
    }
  }
  
  /**
   * Prints a success message
   * @param text Success message
   * @param jsonPath Optional JSON path
   */
  success(text: string, jsonPath?: string): void {
    if (this.options.quiet && this.options.format !== 'json') {
      return;
    }
    
    if (this.options.format === 'text') {
      const indent = this.getIndent();
      this.outputBuffer.push(`${indent}✓ ${this.options.noColor ? text : chalk.green(text)}`);
    } else if (jsonPath && this.options.format === 'json') {
      this.setJsonValue(jsonPath, text);
      this.setJsonValue('success', true);
    }
  }
  
  /**
   * Prints an error message
   * @param text Error message
   * @param jsonPath Optional JSON path
   */
  error(text: string, jsonPath?: string): void {
    if (this.options.format === 'text') {
      const indent = this.getIndent();
      this.outputBuffer.push(`${indent}✖ ${this.options.noColor ? text : chalk.red(text)}`);
    } else if (this.options.format === 'json') {
      this.setJsonValue('error', { message: text });
      if (jsonPath) {
        this.setJsonValue(jsonPath, text);
      }
    }
  }
  
  /**
   * Prints a warning message
   * @param text Warning message
   * @param jsonPath Optional JSON path
   */
  warning(text: string, jsonPath?: string): void {
    if (this.options.quiet && this.options.format !== 'json') {
      return;
    }
    
    if (this.options.format === 'text') {
      const indent = this.getIndent();
      this.outputBuffer.push(`${indent}⚠ ${this.options.noColor ? text : chalk.yellow(text)}`);
    } else if (jsonPath && this.options.format === 'json') {
      this.setJsonValue(jsonPath, text);
      this.setJsonValue('warning', text);
    }
  }
  
  /**
   * Prints an info message
   * @param text Info message
   * @param jsonPath Optional JSON path
   */
  info(text: string, jsonPath?: string): void {
    if (this.options.quiet && this.options.format !== 'json') {
      return;
    }
    
    if (this.options.format === 'text') {
      const indent = this.getIndent();
      this.outputBuffer.push(`${indent}ℹ ${text}`);
    } else if (jsonPath && this.options.format === 'json') {
      this.setJsonValue(jsonPath, text);
    }
  }
  
  /**
   * Prints a verbose message (only in verbose mode)
   * @param text Verbose message
   * @param jsonPath Optional JSON path
   */
  verbose(text: string, jsonPath?: string): void {
    if (!this.options.verbose || (this.options.quiet && this.options.format !== 'json')) {
      return;
    }
    
    if (this.options.format === 'text') {
      const indent = this.getIndent();
      this.outputBuffer.push(`${indent}${this.options.noColor ? text : chalk.dim(text)}`);
    } else if (jsonPath && this.options.format === 'json') {
      if (!this.jsonOutput.verbose) {
        this.jsonOutput.verbose = {};
      }
      this.setJsonValue(`verbose.${jsonPath}`, text);
    }
  }
  
  /**
   * Prints a header
   * @param text Header text
   * @param jsonPath Optional JSON path
   */
  header(text: string, jsonPath?: string): void {
    if (this.options.quiet && this.options.format !== 'json') {
      return;
    }
    
    if (this.options.format === 'text') {
      this.outputBuffer.push('');
      const headerText = this.options.noColor ? text : chalk.bold.yellow(text);
      this.outputBuffer.push(headerText);
      const divider = this.options.noColor ? '─'.repeat(text.length) : chalk.dim('─'.repeat(text.length));
      this.outputBuffer.push(divider);
      this.outputBuffer.push('');
    } else if (jsonPath && this.options.format === 'json') {
      this.setJsonValue(jsonPath, text);
    }
  }
  
  /**
   * Prints a table
   * @param headers Table headers
   * @param rows Table rows
   * @param jsonPath Optional JSON path
   */
  table(headers: string[], rows: string[][], jsonPath?: string): void {
    if (this.options.quiet && this.options.format !== 'json') {
      return;
    }
    
    if (this.options.format === 'text') {
      // Calculate column widths
      const columnWidths: number[] = headers.map((header, i) => {
        const maxRowWidth = Math.max(0, ...rows.map(row => measureTextWidth(row[i] || '')));
        return Math.max(measureTextWidth(header), maxRowWidth);
      });
      
      // Print headers
      const headerRow = headers.map((header, i) => {
        const paddedHeader = header.padEnd(columnWidths[i]);
        return this.options.noColor ? paddedHeader : chalk.bold(paddedHeader);
      }).join(' | ');
      
      this.outputBuffer.push(headerRow);
      
      // Print header divider
      const divider = columnWidths.map(width => '─'.repeat(width)).join('─┼─');
      this.outputBuffer.push(this.options.noColor ? divider : chalk.dim(divider));
      
      // Print rows
      for (const row of rows) {
        const formattedRow = row.map((cell, i) => {
          return (cell || '').padEnd(columnWidths[i]);
        }).join(' | ');
        
        this.outputBuffer.push(formattedRow);
      }
      
      this.outputBuffer.push('');
    } else if (jsonPath && this.options.format === 'json') {
      const tableData = rows.map(row => {
        const obj: Record<string, string> = {};
        headers.forEach((header, i) => {
          obj[header] = row[i] || '';
        });
        return obj;
      });
      
      this.setJsonValue(jsonPath, tableData);
    }
  }
  
  /**
   * Increases the indentation level
   * @returns this for chaining
   */
  indent(): OutputManager {
    this.indentLevel++;
    return this;
  }
  
  /**
   * Decreases the indentation level
   * @returns this for chaining
   */
  outdent(): OutputManager {
    if (this.indentLevel > 0) {
      this.indentLevel--;
    }
    return this;
  }
  
  /**
   * Sets a value in the JSON output
   * @param path Path to the value (dot notation)
   * @param value Value to set
   */
  private setJsonValue(path: string, value: any): void {
    const parts = path.split('.');
    let current = this.jsonOutput;
    
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      if (!current[part]) {
        current[part] = {};
      }
      current = current[part];
    }
    
    current[parts[parts.length - 1]] = value;
  }
  
  /**
   * Flushes the output buffer to stdout
   */
  flush(): void {
    if (this.options.format === 'text') {
      if (this.outputBuffer.length > 0) {
        console.log(this.outputBuffer.join('\n'));
        this.outputBuffer = [];
      }
    } else if (this.options.format === 'json') {
      console.log(JSON.stringify(this.jsonOutput, null, 2));
      this.jsonOutput = {};
    }
  }
  
  /**
   * Clears the output buffer without printing
   */
  clear(): void {
    this.outputBuffer = [];
    this.jsonOutput = {};
  }
  
  /**
   * Gets the current output as a string
   * @returns Output as string
   */
  toString(): string {
    if (this.options.format === 'text') {
      return this.outputBuffer.join('\n');
    } else {
      return JSON.stringify(this.jsonOutput, null, 2);
    }
  }
}

/**
 * Creates a new output manager with current global options
 * @returns New output manager
 */
export function createOutputManager(): OutputManager {
  return new OutputManager(currentOptions);
}

/**
 * Global output manager instance
 */
export const output = createOutputManager();